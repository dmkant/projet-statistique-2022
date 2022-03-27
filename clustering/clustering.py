from cProfile import label
from dataclasses import dataclass
import itertools
from math import ceil
from sys import stdout
from typing import Iterator, List, Union,Tuple

from gensim import models
import numpy as np
import pandas as pd
import word_embedding.distance_wmd as wmd
from pandas import DataFrame
from sklearn.cluster import DBSCAN, KMeans, OPTICS
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.mixture import GaussianMixture as GMM
from sklearn_extra.cluster import KMedoids
import hdbscan as HD
import ClusterEnsembles as ce

seed: int = 1234

def evolution(nActu: int, nFinal: int, rafraichissement: int = 5) -> None:

    facteur: float = 100 / (rafraichissement * nFinal)

    if nActu == 0 or ceil(nActu * facteur) != ceil((nActu - 1) * facteur):
        print(str(ceil(100 * nActu / nFinal)) + '%', end = ' ')
    
    if nActu == nFinal - 1:
        print('\n ')

    stdout.flush()
    

ensembleK: list[int] = [2,3,4,5,6]

# https://scikit-learn-extra.readthedocs.io/en/stable/generated/sklearn_extra.cluster.KMedoids.html
def selection_meilleur_kmedoides(distance: DataFrame, nbRetours: int = None, verbose: bool = True,init_dim=100) -> DataFrame:

    if verbose:
        print("Recherche optimale - K-médoïdes")

    distance = distance.copy().astype('float')

    ensembleinitialisation: list[str] = ['random', 'heuristic', 'k-medoids++', 'build']

    nbModeles: int = len(ensembleK) * len(ensembleinitialisation)

    colonnes = ['K', 'initialisation', 'val_obj', 'silhouette', 'Cal-Harabasz','DBCV']
    resultats = DataFrame(columns = colonnes)

    grille: Iterator = itertools.product(ensembleK, ensembleinitialisation)

    it: int = 0
    for k, methodeInit in grille:

        if verbose:
            evolution(it, nbModeles)

        modele = KMedoids(n_clusters = k, init = methodeInit, metric = 'precomputed', random_state = seed)
        modele.fit(distance)


        silhouette: float = silhouette_score(distance, modele.labels_, metric = 'precomputed')
        calhar: float = None
        DBCV: float = HD.validity_index(np.array(distance).astype(np.float64), modele.labels_,metric='precomputed',d=init_dim)
        resultats.loc[len(resultats.index)] = [k, methodeInit, modele.inertia_, silhouette, calhar,DBCV]

        it += 1

    resultats.sort_values(by = 'silhouette', ascending = False, inplace = True)

    if nbRetours is None:
        nbRetours = len(resultats)

    return resultats.iloc[:nbRetours, :]

# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
def selection_meilleur_kmeans(ev: Union[DataFrame, List[np.ndarray]], nbRetours: int = None, verbose: bool = True) -> DataFrame:
    
    if verbose:
        print("Recherche optimale - K-means")

    # Transformation des données en liste de vecteurs
    # plutôt qu'en dataframe
    if isinstance(ev, DataFrame):
        ev = [ev.iloc[:,i.to_numpy()] for i in range(ev.shape[1])]

    ensembleinitialisation: list[str] = ['k-means++', 'random']
    ensembleTolerance: list[float] = [10**-4]
    nbIter: int = 10

    nbModeles: int = len(ensembleK) * len(ensembleinitialisation) * len(ensembleTolerance)

    colonnes = ['K', 'initialisation', 'nb_iter', 'tolerance', 'val_obj', 'silhouette', 'Cal-Harabasz','DBCV']
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
        DBCV: float = HD.validity_index(np.array(ev).astype(np.float64), modele.labels_)
        resultats.loc[len(resultats.index)] = [k, methodeInit, nbIter, tolerance, modele.inertia_, silhouette, calhar,DBCV]

        it += 1

    resultats.sort_values(by = 'Cal-Harabasz', ascending = False, inplace = True)

    if nbRetours is None:
        nbRetours = len(resultats)

    return resultats.iloc[:nbRetours, :]

# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
def selection_meilleur_dbscan(data: Union[DataFrame, List[np.ndarray]], nbRetours: int = None,
n_jobs: int = 6, verbose: bool = True,init_dim:int = 100,
listeRayons:Union[range,List[float]] = [.25, .5, .75, 1,5,10], listeVoisinage:Union[range,List[int]]=range(2,10),
listeDistances:Union[str,List[str]]=['euclidean', 'manhattan', 'chebyshev']) -> Tuple[DataFrame,List[np.array]]:

    if verbose:
        print("Recherche optimale - DBScan")
    
    if type(listeDistances) not in [list,str]:
        raise ValueError("'listDistances' doit etre une liste")
    elif type(listeDistances) !=  list : 
        listeDistances = [listeDistances]
    
    if "precomputed" in listeDistances and len(listeDistances) > 1:
        raise ValueError("La distance 'precomputed' ne peut pas etre tester en meme temps que les autres ")
    
    if any([arg_type not in [range,list] for arg_type in  [type(listeRayons),type(listeVoisinage)]]):
        raise ValueError("'listeRayons' et 'listeVoisinage' doiventt etre des listes ou des ranges")
    

    if listeDistances == ['precomputed']:
        # Normalisation des données
        data = data.copy() / np.max(data.to_numpy().flat)
    else:
        # Transformation des données en liste de vecteurs
        # plutôt qu'en dataframe
        if isinstance(data, DataFrame):
            data = [data.iloc[:,i.to_numpy()] for i in range(data.shape[1])]

    nbModeles = len(listeVoisinage) * len(listeDistances) * len(listeRayons)
    grille = itertools.product(listeVoisinage, listeDistances, listeRayons)

    colonnes = ['voisinage', 'rayon', 'distance', 'K', 'silhouette', 'Cal-Harabasz','DBCV', 'non_classes']
    resultats = DataFrame(columns = colonnes)
    list_labels = []

    it: int = 0
    for voisinage, distance, rayon in grille:
        metric = 'euclidean' if distance != "precomputed" else "precomputed"

        if verbose:
            evolution(it, nbModeles)

        modele = DBSCAN(eps = rayon, min_samples = voisinage, metric = distance, n_jobs = n_jobs)
        modele.fit(data)
        list_labels.append(np.copy(modele.labels_))

        k: int = len(set(modele.labels_)) - (0 if -1 not in modele.labels_ else 1)
        if k == 1:

            silhouette: float = None
            calhar: float = None
            try:
                DBCV: float = HD.validity_index(np.array(data).astype(np.float64), modele.labels_,d=init_dim,metric=metric)
            except:
                DBCV = None
        else:

            elementsClasses: np.ndarray[int] = np.arange(len(modele.labels_))[modele.labels_ != -1]
            if isinstance(data, DataFrame):

                dataClasses = data.iloc[elementsClasses, elementsClasses] if distance == "precomputed" else data.iloc[:, elementsClasses]

            else:
                dataClasses = data[elementsClasses]
            
            labelsClasses = modele.labels_[elementsClasses]

            silhouette: float = silhouette_score(np.array(dataClasses).astype(np.float64), labelsClasses, metric=metric)
            try:
                DBCV: float = HD.validity_index(np.array(data).astype(np.float64), modele.labels_,d=init_dim,metric=metric)
            except:
                DBCV = None

            if distance != "precomputed":
                calhar: float = calinski_harabasz_score(dataClasses, labelsClasses)

            else:
                calhar: float = None

        calhar, silhouette, DBCV = _evaluation_clustering(labels=modele.labels_,data=data,metric=metric,emb_dim=init_dim)
        resultats.loc[len(resultats.index)] = [voisinage, rayon, distance, len(set(modele.labels_)), silhouette, calhar, DBCV ,np.sum(modele.labels_ == -1)]

        it += 1
    
    list_labels = [list_labels[i] for i in np.argsort(resultats["DBCV"])[::-1]]
    resultats.sort_values(by = 'DBCV', ascending = False, inplace = True)

    if nbRetours is None:
        nbRetours = len(resultats)

    return resultats.iloc[:nbRetours, :], list_labels[:nbRetours]

#https://github.com/scikit-learn-contrib/hdbscan
def selection_meilleur_hdbscan(data: Union[DataFrame, List[np.ndarray]], nbRetours: int = None,
n_jobs: int = 6, verbose: bool = True,init_dim:int = 100,
listeMinClusterSize:Union[range,List[int]]=range(10,50),
listeDistances:Union[str,List[str]]=['euclidean', 'manhattan', 'chebyshev']) -> Tuple[DataFrame,List[np.array]]:

    if verbose:
        print("Recherche optimale - HDBScan")
    
    if type(listeDistances) not in [list,str]:
        raise ValueError("'listDistances' doit etre une liste")
    elif type(listeDistances) !=  list : 
        listeDistances = [listeDistances]
    
    if "precomputed" in listeDistances and len(listeDistances) > 1:
        raise ValueError("La distance 'precomputed' ne peut pas etre tester en meme temps que les autres ")
    
    if type(listeMinClusterSize) not in [range,list]:
        raise ValueError("'listeMinClusterSize' doit etre une liste ou un range")


    if listeDistances != ['precomputed']:
        # Transformation des données en liste de vecteurs
        # plutôt qu'en dataframe
        if isinstance(data, DataFrame):
            data = [data.iloc[:,i.to_numpy()] for i in range(data.shape[1])]


    nbModeles = len(listeMinClusterSize) * len(listeDistances)
    grille = itertools.product(listeMinClusterSize, listeDistances)

    colonnes = ['min_cluster_size', 'distance', 'K', 'silhouette', 'Cal-Harabasz','DBCV', 'non_classes']
    resultats = DataFrame(columns = colonnes)
    list_labels = []

    it: int = 0
    for min_cluster_size, distance in grille:
        metric = "euclidean" if distance != "precomputed" else distance
        if verbose:
            evolution(it, nbModeles)


        clusterer = HD.HDBSCAN(min_cluster_size=min_cluster_size, metric = distance,core_dist_n_jobs= n_jobs)
        cluster_labels = clusterer.fit_predict(np.array(data).astype(np.float64))
        list_labels.append(cluster_labels)

        k: int = len(set(cluster_labels)) - (0 if -1 not in cluster_labels else 1)
        if k == 1:

            silhouette: float = None
            calhar: float = None
            try:
                DBCV: float = HD.validity_index(np.array(data).astype(np.float64), cluster_labels,d=init_dim,metric=metric)
            except:
                DBCV=None

        else:

            elementsClasses: np.ndarray[int] = np.arange(len(cluster_labels))[cluster_labels != -1]
            if isinstance(data, DataFrame):

                dataClasses = data.iloc[elementsClasses, elementsClasses] if distance == 'precomputed' else data.iloc[:, elementsClasses]

            else:
                dataClasses = data[elementsClasses]
            
            labelsClasses = cluster_labels[elementsClasses]

            try:
                silhouette: float = silhouette_score(np.array(dataClasses).astype(np.float64), labelsClasses, metric=metric)
            except Exception as e:
                silhouette = None

            try:
                DBCV: float = HD.validity_index(np.array(data).astype(np.float64), cluster_labels,d=init_dim,metric=metric)
            except:
                DBCV = None

            if distance != 'precomputed':
                calhar: float = calinski_harabasz_score(dataClasses, labelsClasses)

            else:
                calhar: float = None
        calhar, silhouette, DBCV = _evaluation_clustering(labels=cluster_labels,data=data,metric=metric,emb_dim=init_dim)
        resultats.loc[len(resultats.index)] = [min_cluster_size, distance, len(set(cluster_labels)), silhouette, calhar, DBCV ,np.sum(cluster_labels == -1)]

        it += 1

    list_labels = [list_labels[i] for i in np.argsort(resultats["DBCV"])[::-1]]
    resultats.sort_values(by = 'DBCV', ascending = False, inplace = True)

    if nbRetours is None:
        nbRetours = len(resultats)

    return resultats.iloc[:nbRetours, :], list_labels[:nbRetours]

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

    colonnes = ['K', 'covariance', 'silhouette', 'Cal-Harabasz', 'BIC','DBCV']
    resultats = DataFrame(columns = colonnes)
    list_labels = []

    it: int = 0
    for K, covariance in grille:

        if verbose:
            evolution(it, nbModeles)

        modele: GMM = GMM(n_components = K, covariance_type = covariance, n_init = nbIter, random_state = seed)
        labels = modele.fit_predict(ev)
        list_labels.append(labels)

        silhouette: float = silhouette_score(ev, labels)
        calhar: float = calinski_harabasz_score(ev, labels)
        BIC: float = modele.bic(np.array(ev))
        DBCV: float = HD.validity_index(np.array(ev).astype(np.float64), labels)

        resultats.loc[len(resultats.index)] = [K, covariance, silhouette, calhar, BIC, DBCV]

        it += 1

    list_labels = [list_labels[i] for i in np.argsort(resultats["BIC"])]
    resultats.sort_values(by = 'BIC', inplace = True)

    if nbRetours is None:
        nbRetours = len(resultats)

    return resultats.iloc[:nbRetours, :] , list_labels[:nbRetours]

def Ensemble_Clustering(data: Union[DataFrame, List[np.ndarray]], nbRetours: int = None,
type_data: str = 'ev', n_jobs: int = 6, verbose: bool = True,init_dim:int = 100)-> Tuple[DataFrame,np.array]:
    
    resultat, list_labels = selection_meilleur_hdbscan(distances, type_data = 'distance',init_dim=5)
    label_ce = ce.cluster_ensembles(list_labels, solver="hbgf")

def _evaluation_clustering(labels:np.ndarray,data:np.ndarray,metric:str,emb_dim:int=None) -> List[float]:

    elementsClasses: np.ndarray[int] = np.arange(len(labels))[labels != -1]
    if isinstance(data, DataFrame):
        dataClasses = data.iloc[elementsClasses, elementsClasses] if metric == "precomputed" else data.iloc[:, elementsClasses]

    else:
        dataClasses = data[elementsClasses]
    labelsClasses = labels[elementsClasses]

    try:
        DBCV: float = HD.validity_index(np.array(data).astype(np.float64), labels,d=emb_dim,metric=metric)
    except:
        DBCV = None
    
    try:
        silhouette: float = silhouette_score(np.array(dataClasses).astype(np.float64), labelsClasses,metric=metric)
    except Exception as e:
        print(e)
        silhouette = None
    
    if metric != "precomputed":
        try:
            calhar: float = calinski_harabasz_score(dataClasses, labelsClasses)
        except:
            calhar = None
    else:
        calhar = None


    return [calhar, silhouette, DBCV]

if __name__ == '__main__':
    from gensim.models import Word2Vec
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    #Load Modele
    # w2vec: Word2Vec = Word2Vec.load("data/w2vec.bin")
    # ev = [w2vec.wv[v] for v in w2vec.wv.index_to_key]
    ev_cbow = models.KeyedVectors.load_word2vec_format('data/tunning/cbow.kv')
    data = pd.DataFrame(ev_cbow.vectors,index=ev_cbow.index_to_key)
    #Load distance
    distances = wmd.lecture_fichier_distances_wmd('distances_cbow.7z')
    
    # print(selection_meilleur_GMM(ev))
    # print(selection_meilleur_kmeans(ev))
    # print(selection_meilleur_kmedoides(distances))
    # print(selection_meilleur_dbscan(data=distances,init_dim=5,listeDistances="precomputed",listeRayons=[0.25],listeVoisinage=range(10,11,10)))
    print(selection_meilleur_hdbscan(distances, listeDistances = 'precomputed',listeMinClusterSize=range(10,11,10),init_dim=5))
