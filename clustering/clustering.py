import itertools
from math import ceil
from sys import stdout
from typing import Iterator, List, Union

import numpy as np
from pandas import DataFrame
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.mixture import GaussianMixture as GMM

seed: int = 1234


def evolution(nActu: int, nFinal: int, rafraichissement: int = 5) -> None:

    facteur: float = 100 / (rafraichissement * nFinal)

    if nActu == 0 or ceil(nActu * facteur) != ceil((nActu - 1) * facteur):
        print(str(ceil(100 * nActu / nFinal)) + '%', end = ' ')

    
    if nActu == nFinal - 1:
        print('\n ')

    stdout.flush()
    



def selection_meilleur_kmeans(ev: Union[DataFrame, List[np.ndarray]], nbRetours: int = None, verbose: bool = True) -> DataFrame:
    
    if verbose:
        print("Recherche optimale - K-means")

    # Transformation des données en liste de vecteurs
    # plutôt qu'en dataframe
    if isinstance(ev, DataFrame):
        ev = [ev.iloc[:,i.to_numpy()] for i in range(ev.shape[1])]

    ensembleK: list[int] = [2,3,4,5,6]
    ensembleinitialisation: list[str] = ['k-means++', 'random']
    ensembleTolerance: list[float] = [10**-4]
    nbIter: int = 10

    nbModeles: int = len(ensembleK) * len(ensembleinitialisation) * len(ensembleTolerance)

    colonnes = ['K', 'initialisation', 'nb_iter', 'tolerance', 'val_obj', 'silhouette', 'Cal-Harabasz']
    resultats = DataFrame(columns = colonnes)

    grille: Iterator = itertools.product(ensembleK, ensembleinitialisation, ensembleTolerance)

    it: int = 0
    for k, methodeInit, tolerance in grille:

        if verbose:
            evolution(it, nbModeles)

        modele = KMeans(n_clusters = k, init = methodeInit, n_init = nbIter, tol = tolerance, random_state = seed)
        modele.fit(ev)

        
        silhouette: float = silhouette_score(ev, modele.labels_)
        calhar: float = calinski_harabasz_score(ev, modele.labels_)
        resultats.loc[len(resultats.index)] = [k, methodeInit, nbIter, tolerance, modele.inertia_, silhouette, calhar]

        it += 1

    resultats.sort_values(by = 'Cal-Harabasz', ascending = False, inplace = True)

    if nbRetours is None:
        nbRetours = len(resultats)

    return resultats.iloc[:nbRetours, :]



# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
def selection_meilleur_dbscan(ev: Union[DataFrame, List[np.ndarray]], nbRetours: int = None, n_jobs: int = 6, verbose: bool = True) -> DataFrame:

    if verbose:
        print("Recherche optimale - DBScan")

    # Transformation des données en liste de vecteurs
    # plutôt qu'en dataframe
    if isinstance(ev, DataFrame):
        ev = [ev.iloc[:,i.to_numpy()] for i in range(ev.shape[1])]

    listeVoisinage: range = range(3,10)
    listeDistances: list[str] = ['euclidean', 'manhattan', 'chebyshev']
    listeRayons: list[float] = [.25, .5, .75, 1]

    nbModeles = len(listeVoisinage) * len(listeDistances) * len(listeRayons)
    grille = itertools.product(listeVoisinage, listeDistances, listeRayons)

    colonnes = ['voisinage', 'rayon', 'distance', 'K', 'silhouette', 'Cal-Harabasz', 'non_classes']
    resultats = DataFrame(columns = colonnes)

    it: int = 0
    for voisinage, distance, rayon in grille:

        if verbose:
            evolution(it, nbModeles)

        modele = DBSCAN(eps = rayon, min_samples = voisinage, metric = distance, n_jobs = n_jobs)
        modele.fit(ev)

        if len(set(modele.labels_)) == 1 or np.sum(modele.labels_ == -1) > 0:

            silhouette: float = None
            calhar: float = None

        else:

            silhouette: float = silhouette_score(ev, modele.labels_)
            calhar: float = calinski_harabasz_score(ev, modele.labels_)

        resultats.loc[len(resultats.index)] = [voisinage, rayon, distance, len(set(modele.labels_)), silhouette, calhar, np.sum(modele.labels_ == -1)]

        it += 1

    resultats.sort_values(by = 'silhouette', ascending = False, inplace = True)

    if nbRetours is None:
        nbRetours = len(resultats)

    return resultats.iloc[:nbRetours, :]

# https://scikit-learn.org/0.16/modules/generated/sklearn.mixture.GMM.html
def selection_meilleur_GMM(ev: Union[DataFrame, List[np.ndarray]], nbRetours: int = None, verbose: bool = True) -> DataFrame:

    if verbose:
        print("Recherche optimale - GMM")
    # Transformation des données en liste de vecteurs
    # plutôt qu'en dataframe
    if isinstance(ev, DataFrame):
        ev = [ev.iloc[:,i.to_numpy()] for i in range(ev.shape[1])]

    ensembleK: list[int] = [2,3,4,5,6]
    ensembleCovariances: list[str] = ['full', 'tied', 'diag', 'spherical']
    nbIter: int = 10

    nbModeles: int = len(ensembleK) * len(ensembleCovariances)

    grille = itertools.product(ensembleK, ensembleCovariances)

    colonnes = ['K', 'covariance', 'silhouette', 'Cal-Harabasz', 'BIC']
    resultats = DataFrame(columns = colonnes)

    it: int = 0
    for K, covariance in grille:

        if verbose:
            evolution(it, nbModeles)

        modele: GMM = GMM(n_components = K, covariance_type = covariance, n_init = nbIter, random_state = seed)
        labels = modele.fit_predict(ev)

        silhouette: float = silhouette_score(ev, labels)
        calhar: float = calinski_harabasz_score(ev, labels)
        BIC: float = modele.bic(np.array(ev))

        resultats.loc[len(resultats.index)] = [K, covariance, silhouette, calhar, BIC]

        it += 1

    resultats.sort_values(by = 'BIC', inplace = True)

    if nbRetours is None:
        nbRetours = len(resultats)

    return resultats.iloc[:nbRetours, :]



if __name__ == '__main__':
    from gensim.models import Word2Vec


    w2vec: Word2Vec = Word2Vec.load("data/w2vec.bin")
    # espace vectoriel généré des mots
    ev = [w2vec.wv[v] for v in w2vec.wv.index_to_key]
    print(selection_meilleur_GMM(ev))
    print(selection_meilleur_kmeans(ev))
    print(selection_meilleur_dbscan(ev))
