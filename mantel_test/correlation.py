from itertools import permutations
from time import time
import numpy as np
from scipy import spatial, stats

from joblib import Parallel, delayed
import multiprocessing
from multiprocessing import Process


def test(X, Y, perms=10000, method="pearson", tail="two-tail",min_corr=1,ignore_nans=False, parallel=False,n_jobs=1):
    """
    Takes two distance matrices (either redundant matrices or condensed vectors)
    and performs a Mantel test. The Mantel test is a significance test of the
    correlation between two distance matrices.
    Parameters
    ----------
    X : array_like
            First distance matrix (condensed or redundant).
    Y : array_like
            Second distance matrix (condensed or redundant), where the order of
            elements corresponds to the order of elements in the first matrix.
    perms : int, optional
            The number of permutations to perform (default: 10000). A larger
            number gives more reliable results but takes longer to run. If the
            number of possible permutations is smaller, all permutations will
            be tested. This can be forced by setting perms to 0.
    method : str, optional
            Type of correlation coefficient to use; either 'pearson' or 'spearman'
            (default: 'pearson').
    tail : str, optional
            Which tail to test in the calculation of the empirical p-value; either
            'upper', 'lower', or 'two-tail' (default: 'two-tail').
    ignore_nans : bool, optional
            Ignore NaN values in the Y matrix (default: False). This can be
            useful if you have missing values in one of the matrices.
    parallel : bool, optional
            Wether use parallel computing or not.
    n_jobs : int, optional
            Number of job for parallel computing.
    
    Returns
    -------
    r : float
            Veridical correlation
    p : float
            Empirical p-value
    z : float
            Standard score (z-score)
    """

    # Ensure that X and Y are represented as Numpy arrays.
    X = np.asarray(X)
    Y = np.asarray(Y)

    # Check that X and Y are valid distance matrices.
    if (
        spatial.distance.is_valid_dm(np.nan_to_num(X)) == False
        and spatial.distance.is_valid_y(X) == False
    ):
        raise ValueError("X is not a valid condensed or redundant distance matrix")
    if (
        spatial.distance.is_valid_dm(np.nan_to_num(Y)) == False
        and spatial.distance.is_valid_y(Y) == False
    ):
        raise ValueError("Y is not a valid condensed or redundant distance matrix")

    # If X or Y is a redundant distance matrix, reduce it to a condensed distance matrix.
    if len(X.shape) == 2:
        X = spatial.distance.squareform(X, force="tovector", checks=False)
    if len(Y.shape) == 2:
        Y = spatial.distance.squareform(Y, force="tovector", checks=False)

    # Check for size equality.
    if len(X) != len(Y):
        raise ValueError("X and Y are not of equal size")

    # Check for minimum size.
    if len(X) < 3:
        raise ValueError("X and Y should represent at least 3 objects")

    # Check finiteness of X and Y
    if not np.isfinite(X).all():
        raise ValueError(
            "X cannot contain NaNs (but Y may contain NaNs, so consider reordering X and Y)"
        )
    finite_Y = np.isfinite(Y)
    if not ignore_nans and not finite_Y.all():
        raise ValueError('Y may contain NaNs, but "ignore_nans" must be set to True')
    if ignore_nans and finite_Y.all():
        ignore_nans = False  # ignore_nans is True but Y contains no nans

    # If Spearman correlation is requested, convert X and Y to ranks.
    method = method.lower()
    if method == "spearman":
        X, Y = stats.rankdata(X), stats.rankdata(Y)
        Y[~finite_Y] = np.nan  # retain any nans, so that these can be ignored later

    # Check for valid method parameter.
    elif method != "pearson":
        raise ValueError('The method should be set to "pearson" or "spearman"')

    # Check for valid tail parameter.
    tail = tail.lower()
    if tail not in ["upper", "lower", "two-tail"]:
        raise ValueError('The tail should be set to "upper", "lower", or "two-tail"')

    # Now we're ready to start the Mantel test using a number of optimizations:
    #
    # 1. Rather than compute correlation coefficients, we'll just compute the
    #    covariances. This works because the denominator in the equation for the
    #    correlation coefficient will yield the same result however the objects
    #    are permuted, making it redundant. Removing the denominator leaves us
    #    with the covariance.
    #
    # 2. Rather than permute the Y distances and derive the residuals to calculate
    #    the covariance with the X distances, we'll represent the Y residuals in
    #    the matrix and shuffle those directly.
    #
    # 3. If the number of possible permutations is less than the number of
    #    permutations that were requested, we'll run a deterministic test where
    #    we try all possible permutations rather than sample the permutation
    #    space. This gives a faster, deterministic result.

    # Calculate the X and Y residuals, which will be used to compute the
    # covariance under each permutation.
    X_residuals = X - np.mean(X[finite_Y])
    Y_residuals = Y - np.mean(Y[finite_Y])

    # Expand the Y residuals to a redundant matrix.
    Y_residuals_as_matrix = spatial.distance.squareform(
        Y_residuals, force="tomatrix", checks=False
    )

    m = len(Y_residuals_as_matrix)  # number of objects
    n = np.math.factorial(m)  # number of possible matrix permutations

    # If the number of requested permutations is greater than the number of
    # possible permutations (m!) or the perms parameter is set to 0, then run a
    # deterministic Mantel test
    if perms >= n or perms == 0:
        if ignore_nans:
            correlations = deterministic_test_with_nans(m, n, X, Y_residuals_as_matrix)
        else:
            correlations = deterministic_test(m, n, X_residuals, Y_residuals_as_matrix)
        # correlations[0] is the veridical correlation

    else:
        if ignore_nans:
            correlations = stochastic_test_with_nans(m, perms, X, Y_residuals_as_matrix)
        else:
            if parallel:
                correlations = stochastic_test_parallel(m, perms, X_residuals, Y_residuals_as_matrix,n_jobs=n_jobs,min_cov = min_corr)
            else:
                correlations = stochastic_test(m, perms, X_residuals, Y_residuals_as_matrix)
        correlations[0] = sum(X_residuals[finite_Y] * Y_residuals[finite_Y]) / np.sqrt(
            sum(X_residuals[finite_Y] ** 2) * sum(Y_residuals[finite_Y] ** 2)
        )  # compute veridical correlation and place in positon 0

    r = correlations[0]

    if tail == "upper":
        p = sum(correlations >= r) / len(correlations)
    elif tail == "lower":
        p = sum(correlations <= r) / len(correlations)
    elif tail == "two-tail":
        p = sum(abs(correlations) >= abs(r)) / len(correlations)

    z = (r - np.mean(correlations)) / np.std(correlations)

    return r, p, z,correlations


def deterministic_test(m, n, X_residuals, Y_residuals_as_matrix):
    Y_residuals_permuted = np.zeros((m ** 2 - m) // 2)
    covariances = np.zeros(n)
    for i, order in enumerate(permutations(range(m))):
        Y_residuals_as_matrix_permuted = Y_residuals_as_matrix[order, :][:, order]
        spatial.distance._distance_wrap.to_vector_from_squareform_wrap(
            Y_residuals_as_matrix_permuted, Y_residuals_permuted
        )
        covariances[i] = (X_residuals * Y_residuals_permuted).sum()
    denominator = np.sqrt(sum(X_residuals ** 2) * sum(Y_residuals_permuted ** 2))
    return covariances / denominator


def deterministic_test_with_nans(m, n, X, Y_residuals_as_matrix):
    Y_residuals_permuted = np.zeros((m ** 2 - m) // 2)
    correlations = np.zeros(n)
    for i, order in enumerate(permutations(range(m))):
        Y_residuals_as_matrix_permuted = Y_residuals_as_matrix[order, :][:, order]
        spatial.distance._distance_wrap.to_vector_from_squareform_wrap(
            Y_residuals_as_matrix_permuted, Y_residuals_permuted
        )
        # Since on each permutation we will be ignoring different values in X,
        # the X_residuals need to be recomputed each time depending on which
        # values in permuted Y are finite.
        finite_Y_permuted = np.isfinite(Y_residuals_permuted)
        reduced_X = X[finite_Y_permuted]
        reduced_X_residuals = reduced_X - reduced_X.mean()
        reduced_Y_residuals = Y_residuals_permuted[finite_Y_permuted]
        covariance = (reduced_X_residuals * reduced_Y_residuals).sum()
        # The denominator will be different on each permutation
        denominator = np.sqrt(
            sum(reduced_X_residuals ** 2) * sum(reduced_Y_residuals ** 2)
        )
        correlations[i] = covariance / denominator
    return correlations


def stochastic_test(m, n, X_residuals, Y_residuals_as_matrix):
    Y_residuals_permuted = np.zeros((m ** 2 - m) // 2)
    covariances = np.zeros(n)
    order = np.arange(m)
    t = time()
    for i in range(1, n):
        np.random.shuffle(order)
        Y_residuals_as_matrix_permuted = Y_residuals_as_matrix[order, :][:, order]
        spatial.distance._distance_wrap.to_vector_from_squareform_wrap(
            Y_residuals_as_matrix_permuted, Y_residuals_permuted
        )
        covariances[i] = (X_residuals * Y_residuals_permuted).sum()
    print(f"loop time {time() - t}")
    t = time()
    denominator = np.sqrt(sum(X_residuals ** 2) * sum(Y_residuals_permuted ** 2))
    print(f"denominateur time {time() - t}")
    return covariances / denominator

# A global dictionary storing the variables passed from the initializer.
var_dict = {}

def init_worker(X_residuals, X_residuals_shape, Y_residuals_as_matrix, Y_residuals_as_matrix_shape, Y_residuals_permuted, Y_residuals_permuted_shape):
    var_dict['X_residuals'] = X_residuals
    var_dict['X_residuals_shape'] = X_residuals_shape
    
    var_dict['Y_residuals_as_matrix'] = Y_residuals_as_matrix
    var_dict['Y_residuals_as_matrix_shape'] = Y_residuals_as_matrix_shape
    
    var_dict['Y_residuals_permuted'] = Y_residuals_permuted
    var_dict['Y_residuals_permuted_shape'] = Y_residuals_permuted_shape

def compute_covariance(order,min_cov):
    # print(f"debut calcul {i}")
    X_np = np.frombuffer(var_dict['X_residuals']).reshape(var_dict['X_residuals_shape'])
    Y_residuals_permuted = np.frombuffer(var_dict['Y_residuals_permuted']).reshape(var_dict['Y_residuals_permuted_shape'])
    Y_residuals_as_matrix = np.frombuffer(var_dict['Y_residuals_as_matrix']).reshape(var_dict['Y_residuals_as_matrix_shape'])
    
    covariance = 0
    while(np.abs(covariance) >= min_cov):
        np.random.shuffle(order)
        Y_residuals_as_matrix_permuted = Y_residuals_as_matrix[order, :][:, order]
        spatial.distance._distance_wrap.to_vector_from_squareform_wrap(
            Y_residuals_as_matrix_permuted, Y_residuals_permuted
        )
        covariance = (X_np * Y_residuals_permuted).sum()
    # print(f"fin calcul {i} : covariance: {covariance} -- {X_np[:5]}...{Y_residuals_permuted[:5]}....")
    return covariance


def stochastic_test_parallel(m, n, X_residuals, Y_residuals_as_matrix,min_cov=0,n_jobs=2):
    Y_residuals_permuted = np.zeros((m ** 2 - m) // 2)
    orders = [np.arange(m) for _ in range(n)]
    
    X_residuals_shared = multiprocessing.RawArray('d', X_residuals.shape[0])
    Y_residuals_as_matrix_shared = multiprocessing.RawArray('d', Y_residuals_as_matrix.shape[0]*Y_residuals_as_matrix.shape[1])
    Y_residuals_permuted_shared = multiprocessing.RawArray('d', Y_residuals_permuted.shape[0])

    X_residuals_np = np.frombuffer(X_residuals_shared).reshape(X_residuals.shape)
    Y_residuals_as_matrix_np = np.frombuffer(Y_residuals_as_matrix_shared).reshape(Y_residuals_as_matrix.shape)
    Y_residuals_permuted_np = np.frombuffer(Y_residuals_permuted_shared).reshape(Y_residuals_permuted.shape)

    np.copyto(X_residuals_np, X_residuals)
    np.copyto(Y_residuals_as_matrix_np, Y_residuals_as_matrix)
    np.copyto(Y_residuals_permuted_np, Y_residuals_permuted)
    
    
    # covariances = [0] + Parallel(n_jobs=n_jobs)(delayed(compute_covariance)(orders[i],Y_residuals_permuted, Y_residuals_as_matrix,X_residuals) for i in range(1,n))
    with multiprocessing.Pool(initializer=init_worker, initargs=(X_residuals_shared, X_residuals.shape, 
                                                                 Y_residuals_as_matrix_shared, Y_residuals_as_matrix.shape, 
                                                                 Y_residuals_permuted_shared, Y_residuals_permuted.shape)) as pool:
        results = [pool.apply_async(compute_covariance, (orders[i],min_cov)) for i in range(1,n)]
        covariances = [0] + [f.get() for f in results]
    
    spatial.distance._distance_wrap.to_vector_from_squareform_wrap(
        Y_residuals_as_matrix, Y_residuals_permuted
    )
    denominator = np.sqrt(sum(X_residuals ** 2) * sum(Y_residuals_permuted ** 2))
    return covariances / denominator


def stochastic_test_with_nans(m, n, X, Y_residuals_as_matrix):
    Y_residuals_permuted = np.zeros((m ** 2 - m) // 2)
    correlations = np.zeros(n)
    order = np.arange(m)
    for i in range(1, n):
        np.random.shuffle(order)
        Y_residuals_as_matrix_permuted = Y_residuals_as_matrix[order, :][:, order]
        spatial.distance._distance_wrap.to_vector_from_squareform_wrap(
            Y_residuals_as_matrix_permuted, Y_residuals_permuted
        )
        # Since on each permutation we will be ignoring different values in X,
        # the X_residuals need to be recomputed each time depending on which
        # values in permuted Y are finite.
        finite_Y_permuted = np.isfinite(Y_residuals_permuted)
        reduced_X = X[finite_Y_permuted]
        reduced_X_residuals = reduced_X - reduced_X.mean()
        reduced_Y_residuals = Y_residuals_permuted[finite_Y_permuted]
        covariance = (reduced_X_residuals * reduced_Y_residuals).sum()
        # The denominator will be different on each permutation
        denominator = np.sqrt(
            sum(reduced_X_residuals ** 2) * sum(reduced_Y_residuals ** 2)
        )
        correlations[i] = covariance / denominator
    return correlations