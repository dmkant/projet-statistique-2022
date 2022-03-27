from cProfile import label
from dataclasses import dataclass
import itertools
from math import ceil
from sys import stdout
from typing import Iterator, List, Union,Tuple
import json

from gensim import models
import numpy as np
import pandas as pd
from sklearn import ensemble
import word_embedding.distance_wmd as wmd
import doc_embedding.moyenne as moy_emb
from pandas import DataFrame
from sklearn.cluster import DBSCAN, KMeans, OPTICS
from sklearn.metrics import calinski_harabasz_score, euclidean_distances, silhouette_score, normalized_mutual_info_score
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
    



# https://scikit-learn-extra.readthedocs.io/en/stable/generated/sklearn_extra.cluster.KMedoids.html
def selection_meilleur_kmedoides(
    distance: DataFrame, 
    nbRetours: int = None, 
    verbose: bool = True,
    init_dim=100,
    ensembleK: List[int] = [2,3,4,5,6],
    ensembleinitialisation: List[str] = ['random', 'heuristic', 'k-medoids++', 'build']) -> Tuple[DataFrame,np.ndarray]:

    if verbose:
        print("Recherche optimale - K-médoïdes")

    distance = distance.copy().astype('float')

    nbModeles: int = len(ensembleK) * len(ensembleinitialisation)

    colonnes = ['K', 'initialisation', 'val_obj', 'silhouette', 'Cal-Harabasz','DBCV']
    resultats = DataFrame(columns = colonnes)
    list_labels = []

    grille: Iterator = itertools.product(ensembleK, ensembleinitialisation)

    it: int = 0
    for k, methodeInit in grille:

        if verbose:
            evolution(it, nbModeles)

        modele = KMedoids(n_clusters = k, init = methodeInit, metric = 'precomputed', random_state = seed)
        modele.fit(distance)
        list_labels.append(np.copy(modele.labels_))


        silhouette: float = silhouette_score(distance, modele.labels_, metric = 'precomputed')
        calhar: float = None
        DBCV: float = HD.validity_index(np.array(distance).astype(np.float64), modele.labels_,metric='precomputed',d=init_dim)
        calhar, silhouette, DBCV = _evaluation_clustering(labels=modele.labels_,data=distance,metric='precomputed',emb_dim=init_dim)
        resultats.loc[len(resultats.index)] = [k, methodeInit, modele.inertia_, silhouette, calhar,DBCV]

        it += 1

    list_labels = [list_labels[i] for i in np.argsort(resultats['silhouette'])[::-1]]
    resultats.sort_values(by = 'silhouette', ascending = False, inplace = True)

    if nbRetours is None:
        nbRetours = len(resultats)

    return resultats.iloc[:nbRetours, :], np.array(list_labels[:nbRetours])

# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
def selection_meilleur_kmeans(
    ev: Union[DataFrame, List[np.ndarray]],
    nbRetours: int = None, verbose: bool = True,
    ensembleK: List[int] = [2,3,4,5,6],
    ensembleinitialisation: List[str] = ['k-means++', 'random']) -> Tuple[DataFrame,np.ndarray]:
    
    if verbose:
        print("Recherche optimale - K-means")

    # Transformation des données en liste de vecteurs
    # plutôt qu'en dataframe
    if isinstance(ev, DataFrame):
        ev = [ev.iloc[:,i.to_numpy()] for i in range(ev.shape[1])]

    ensembleTolerance: list[float] = [10**-4]
    nbIter: int = 10

    nbModeles: int = len(ensembleK) * len(ensembleinitialisation) * len(ensembleTolerance)

    colonnes = ['K', 'initialisation', 'nb_iter', 'tolerance', 'val_obj', 'silhouette', 'Cal-Harabasz','DBCV']
    resultats = DataFrame(columns = colonnes)
    list_labels = []

    grille: Iterator = itertools.product(ensembleK, ensembleinitialisation, ensembleTolerance)

    it: int = 0
    for k, methodeInit, tolerance in grille:

        if verbose:
            evolution(it, nbModeles)

        modele = KMeans(n_clusters = k, init = methodeInit, n_init = nbIter, tol = tolerance, random_state = seed)
        modele.fit(ev)
        list_labels.append(np.copy(modele.labels_))

        calhar, silhouette, DBCV = _evaluation_clustering(labels=modele.labels_,data=ev)
        resultats.loc[len(resultats.index)] = [k, methodeInit, nbIter, tolerance, modele.inertia_, silhouette, calhar,DBCV]

        it += 1
    
    list_labels = [list_labels[i] for i in np.argsort(resultats['Cal-Harabasz'])[::-1]]
    resultats.sort_values(by = 'Cal-Harabasz', ascending = False, inplace = True)

    if nbRetours is None:
        nbRetours = len(resultats)

    return resultats.iloc[:nbRetours, :], np.array(list_labels[:nbRetours])

# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
def selection_meilleur_dbscan(
    data: Union[DataFrame, List[np.ndarray]],
    nbRetours: int = None,
    n_jobs: int = 6, 
    verbose: bool = True,
    init_dim:int = 100,
    listeRayons:Union[range,List[float]] = [.25, .5, .75, 1,5,10],
    listeVoisinage:Union[range,List[int]]=range(2,10),
    listeDistances:Union[str,List[str]]=['euclidean', 'manhattan', 'chebyshev']) -> Tuple[DataFrame,np.ndarray]:

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

        calhar, silhouette, DBCV = _evaluation_clustering(labels=modele.labels_,data=data,metric=metric,emb_dim=init_dim)
        resultats.loc[len(resultats.index)] = [voisinage, rayon, distance, len(set(modele.labels_)), silhouette, calhar, DBCV ,np.sum(modele.labels_ == -1)]

        it += 1
    
    list_labels = [list_labels[i] for i in np.argsort(resultats["DBCV"])[::-1]]
    resultats.sort_values(by = 'DBCV', ascending = False, inplace = True)

    if nbRetours is None:
        nbRetours = len(resultats)

    return resultats.iloc[:nbRetours, :], np.array(list_labels[:nbRetours])

#https://github.com/scikit-learn-contrib/hdbscan
def selection_meilleur_hdbscan(
    data: Union[DataFrame, List[np.ndarray]],
    nbRetours: int = None,
    n_jobs: int = 6,
    verbose: bool = True,init_dim:int = 100,
    listeMinClusterSize:Union[range,List[int]]=range(10,50),
    listeDistances:Union[str,List[str]]=['euclidean', 'manhattan', 'chebyshev']) -> Tuple[DataFrame,np.ndarray]:

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

        calhar, silhouette, DBCV = _evaluation_clustering(labels=cluster_labels,data=data,metric=metric,emb_dim=init_dim)
        resultats.loc[len(resultats.index)] = [min_cluster_size, distance, len(set(cluster_labels)), silhouette, calhar, DBCV ,np.sum(cluster_labels == -1)]

        it += 1

    list_labels = [list_labels[i] for i in np.argsort(resultats["DBCV"])[::-1]]
    resultats.sort_values(by = 'DBCV', ascending = False, inplace = True)

    if nbRetours is None:
        nbRetours = len(resultats)

    return resultats.iloc[:nbRetours, :], np.array(list_labels[:nbRetours])

# https://scikit-learn.org/0.16/modules/generated/sklearn.mixture.GMM.html
def selection_meilleur_GMM(
    ev: Union[DataFrame, List[np.ndarray]],
    nbRetours: int = None,
    verbose: bool = True,
    ensembleK: List[int] = [2,3,4,5,6],
    ensembleCovariances: List[str] = ['full', 'tied', 'diag', 'spherical']) -> Tuple[DataFrame,np.ndarray]:

    if verbose:
        print("Recherche optimale - GMM")
    # Transformation des données en liste de vecteurs
    # plutôt qu'en dataframe
    if isinstance(ev, DataFrame):
        ev = [ev.iloc[:,i.to_numpy()] for i in range(ev.shape[1])]

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

        BIC: float = modele.bic(np.array(ev))
        calhar, silhouette, DBCV = _evaluation_clustering(labels=labels,data=ev)

        resultats.loc[len(resultats.index)] = [K, covariance, silhouette, calhar, BIC, DBCV]

        it += 1

    list_labels = [list_labels[i] for i in np.argsort(resultats["BIC"])]
    resultats.sort_values(by = 'BIC', inplace = True)

    if nbRetours is None:
        nbRetours = len(resultats)

    return resultats.iloc[:nbRetours, :] , np.array(list_labels[:nbRetours])

def Ensemble_Clustering(
    data: Union[DataFrame, List[np.ndarray]], 
    distance:Union[DataFrame, List[np.ndarray]]=None, 
    nbRetours: int = None,
    n_jobs: int = 6, 
    verbose: bool = True,
    init_dim:int = 100,
    listAlgo:List[str] = ["kmeans","gmm","hdbscan"],
    ensembleK: List[int] = [2, 3, 4, 5, 6], 
    ensembleCovariances: List[str] = ['full', 'tied', 'diag', 'spherical'],
    listeMinClusterSize: Union[range , List[int]] = range(10, 50), 
    listeDistances: Union[str ,List[str]] = ['euclidean', 'manhattan', 'chebyshev'],
    listSolver:Union[List[str],str] = "hbgf",
    listeK: List[int]=[None])-> Tuple[DataFrame,np.ndarray]:
    
    if type(listeDistances) not in [list,str]:
        raise ValueError("'listDistances' doit etre une liste")
    elif type(listeDistances) !=  list : 
        listeDistances = [listeDistances]

    if "precomputed" in listeDistances and distance is None:
        raise ValueError("La distance 'precomputed' necessite le parametre distance ")

    if type(listSolver) not in [list,str]:
        raise ValueError("'listSolver' doit etre une liste")
    elif type(listSolver) !=  list : 
        listSolver = [listSolver]
    if any([solver not in ['cspa','hgpa','mcla','hbgf','nmf'] for solver in listSolver]):
        raise ValueError("'listSolver' doit etre contenu dans ['cspa','hgpa','mcla','hbgf','nmf']") 

    if  len(listAlgo) == 0:
        raise ValueError("'listAlgo' doit etre une liste contenant au moins un des elements : ['kmeans','gmm','hdbscan']")


    if verbose:
        print("Recherche optimale - Concensus Clustering")
        print("Entrainement des algo de clustering")
    
    allLabels = []
    if 'kmeans' in listAlgo:
        resultatKmean, listLabelsKmeans = selection_meilleur_kmeans(data,nbRetours=nbRetours,verbose=verbose,ensembleK=ensembleK)
        allLabels.append(listLabelsKmeans)
        if verbose:
            print('Kmeans')
            print(resultatKmean)
    
    if 'gmm' in listAlgo:
        resultatGMM, listLabelsGMM = selection_meilleur_GMM(data,nbRetours=nbRetours,verbose=verbose,ensembleK=ensembleK,ensembleCovariances=ensembleCovariances)
        allLabels.append(listLabelsGMM)
        if verbose:
            print('GMM')
            print(resultatGMM)    
    
    if 'hdbscan' in listAlgo:
        #a ameliorer
        if "precomputed" in listeDistances:
            resultatHdbscan, listLabelsHdbscan = selection_meilleur_hdbscan(distance,nbRetours=nbRetours,verbose=verbose, n_jobs=n_jobs,init_dim=init_dim,listeDistances="precomputed",listeMinClusterSize=listeMinClusterSize)
        else:
            resultatHdbscan, listLabelsHdbscan = selection_meilleur_hdbscan(data,nbRetours=nbRetours,verbose=verbose, n_jobs=n_jobs,init_dim=init_dim,listeDistances=listeDistances,listeMinClusterSize=listeMinClusterSize)
        
        allLabels.append(listLabelsHdbscan)
        if verbose:
            print('HDBSCAN')
            print(resultatHdbscan)

    allLabels = np.concatenate(allLabels,axis=0)

    nbModeles: int = len(listeK) * len(listSolver)
    colonnes = ['K', 'Solver','silhouette', 'Cal-Harabasz','DBCV','ANMI']
    resultatsCe = DataFrame(columns = colonnes)
    list_labels = []

    grille = itertools.product(listeK, listSolver)
    
    it: int = 0
    for K, solver in grille:

        if verbose:
            evolution(it, nbModeles)

        labelCe = ce.cluster_ensembles(allLabels, solver=solver,nclass=K)
        list_labels.append(labelCe)

        K = len(set(labelCe))
        calhar, silhouette, DBCV = _evaluation_clustering(labels=labelCe,data=data)
        ANMI = np.mean([normalized_mutual_info_score(labels_true=labels,labels_pred=labelCe) for labels in allLabels])

        resultatsCe.loc[len(resultatsCe.index)] = [K,solver, silhouette, calhar, DBCV,ANMI]

        it += 1

    return resultatsCe, np.array(list_labels)


def _evaluation_clustering(
    labels:np.ndarray,
    data:np.ndarray,
    metric:str="euclidean",
    emb_dim:int=None) -> List[float]:


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
    #Load distance
    # distances = wmd.lecture_fichier_distances_wmd('distances_cbow.7z')
    
    # print(selection_meilleur_GMM(ev_cbow.vectors))
    # print(selection_meilleur_kmeans(ev_cbow.vectors))
    # print(selection_meilleur_kmedoides(distances))
    # print(selection_meilleur_dbscan(data=distances,init_dim=5,listeDistances="precomputed",listeRayons=[0.25],listeVoisinage=range(10,11,10)))
    # print(selection_meilleur_hdbscan(distances, listeDistances = 'precomputed',listeMinClusterSize=range(10,101,10),init_dim=5))

    ev_cbow = models.KeyedVectors.load_word2vec_format('data/tunning/cbow.kv')
    with open("data/docs.json") as file:
        docs = json.load(file)
    moy_embedding_tfidf = moy_emb.word_emb_vers_doc_emb_moyenne(docs=docs, modele=ev_cbow, methode = 'TF-IDF')
    distances = euclidean_distances(moy_embedding_tfidf)

    print(Ensemble_Clustering(
        data=moy_embedding_tfidf,
        distance=pd.DataFrame(distances),
        listeDistances="precomputed", 
        nbRetours=5,
        listSolver=["mcla","hbgf"],
        init_dim=moy_embedding_tfidf.shape[1])
        )
