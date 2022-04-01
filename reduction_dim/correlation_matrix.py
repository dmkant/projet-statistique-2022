import numpy as np
import scipy.stats

def flat_matrix(X_mat:np.ndarray):
    flat_index = np.triu_indices(X_mat.shape[0])
    X_mat_flat = X_mat[flat_index]

    return X_mat_flat

def correlation_epsilon(
    initial_distance: np.array,
    new_distance: np.array,
    epsilon: float = 100,
    type: str = "pearson",
) -> float:
    if initial_distance.shape != new_distance.shape:
        raise "Les matrices doivent avoir la meme dimension"
    if len(initial_distance.shape) != 2:
        raise "initial_distance et new_distan ce doivent être 2D"
    if initial_distance.shape[0] != initial_distance.shape[1]:
        raise "initial_distance et new_distance doivent être carre"
    if type not in ["pearson", "spearman"]:
        raise "type doit etre 'pearson' ou 'spearman'"

    flat_index = np.triu_indices(initial_distance.shape[0])

    initial_distance_flat = initial_distance[flat_index]
    new_distance_flat = new_distance[flat_index]

    if type == "pearson":
        coeff = scipy.stats.pearsonr(
            initial_distance_flat[np.where(initial_distance_flat < epsilon)],
            new_distance_flat[np.where(initial_distance_flat < epsilon)],
        )[0]
    else:
        coeff = scipy.stats.spearmanr(
            initial_distance_flat[np.where(initial_distance_flat < epsilon)],
            new_distance_flat[np.where(initial_distance_flat < epsilon)],
        )[0]

    return coeff


def correlation_neighboor(
    initial_distance: np.array,
    new_distance: np.array,
    k: int = 100,
    type: str = "pearson",
) -> float:
    if initial_distance.shape != new_distance.shape:
        raise "Les matrices doivent avoir la meme dimension"
    if len(initial_distance.shape) != 2:
        raise "initial_distance et new_distance doivent être 2D"
    if initial_distance.shape[0] != initial_distance.shape[1]:
        raise "initial_distance et new_distance doivent être carre"
    if type not in ["pearson", "spearman"]:
        raise "type doit etre 'pearson' ou 'spearman'"

    flat_index = np.triu_indices(initial_distance.shape[0])

    sup_value = np.max(initial_distance) + 1
    initial_distance_neighbor = np.copy(initial_distance)
    # les indices
    longest_neighbor_index = np.apply_along_axis(np.argsort, 1, initial_distance)[
        :, ::-1
    ][:, :k]
    for i in range(initial_distance.shape[0]):
        initial_distance[i, longest_neighbor_index[i, :]] = sup_value

    initial_distance_flat = initial_distance[flat_index]
    new_distance_flat = new_distance[flat_index]

    if type == "pearson":
        coeff = scipy.stats.pearsonr(
            initial_distance_flat[np.where(initial_distance_flat < sup_value)],
            new_distance_flat[np.where(initial_distance_flat < sup_value)],
        )[0]
    else:
        coeff = scipy.stats.spearmanr(
            initial_distance_flat[np.where(initial_distance_flat < sup_value)],
            new_distance_flat[np.where(initial_distance_flat < sup_value)],
        )[0]

    return coeff
