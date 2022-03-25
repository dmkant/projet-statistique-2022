from operator import mod
from sklearn.manifold import MDS
import pandas as pd
import time
import multiprocessing




from sklearn.metrics.pairwise import euclidean_distances
import scipy.spatial as sp
import scipy.stats
import random
import math

from word_embedding.distance_wmd import *
from reduction_dim.correlation_matrix import *


# A global dictionary storing the variables passed from the initializer.
var_dict = {}

def init_worker(mat_distance_wmd, mat_distance_wmd_shape,model):
    var_dict['mat_distance_wmd'] = mat_distance_wmd
    var_dict['mat_distance_wmd_shape'] = mat_distance_wmd_shape
    var_dict['model'] = model
    

def MDS_evaluation(dim):
    distance_wmd_np = np.frombuffer(var_dict['mat_distance_wmd']).reshape(var_dict['mat_distance_wmd_shape'])
    model = var_dict['model']

    mds_model = MDS(n_components=dim, dissimilarity="precomputed",n_jobs=-1)
    print(f"{model}:{dim} fit MDS")
    t0 = time.time()
    mds_embedding = mds_model.fit_transform(np.array(distance_wmd_np))
    t1 = time.time() - t0
    mds_distance = euclidean_distances(mds_embedding)
    print(f"{model}:{dim} temps: {t1} calcul correlation")
    r_pearson = correlation_epsilon(distance_wmd_np,mds_distance,epsilon=np.inf,type="pearson")
    r_spearman = correlation_epsilon(distance_wmd_np,mds_distance,epsilon=np.inf,type="spearman")
    
    print([model, dim, r_pearson,r_spearman, mds_model.stress_,t1])    
    return [model, dim, r_pearson,r_spearman, mds_model.stress_,t1]

def MDS_parallel(word_embedding_model,n_jobs=2):

    for model in word_embedding_model:
        print("lecture")
        mat_distance_wmd = np.array(lecture_fichier_distances_wmd(f"distances_{model}.7z"))
        mat_distance_wmd_shared = multiprocessing.RawArray('d', mat_distance_wmd.shape[0]**2)
        mat_distance_wmd_np = np.frombuffer(mat_distance_wmd_shared).reshape(mat_distance_wmd.shape)
        np.copyto(mat_distance_wmd_np, mat_distance_wmd)        
        
        with multiprocessing.Pool(initializer=init_worker, initargs=(mat_distance_wmd_shared, mat_distance_wmd.shape, model)) as pool:
            results = [pool.apply_async(MDS_evaluation,(dim,)) for dim in range(1500,10001,500) ]
            df_evaluation = pd.DataFrame([f.get() for f in results],
                                         columns=[ "model", "dim", "corr_p","corr_s","stress","time"])
            df_evaluation.to_csv(f"data/tunning/MDS/mds_wmd_{model}2.csv",sep=";",index=False)


use_parrallel = True
word_embedding_model = ["cbow","skipgram","glove"]

if use_parrallel:
    MDS_parallel(word_embedding_model)    
else:
    list_model = []
    list_dim = []
    list_corr_p = []
    list_corr_s = []
    list_stress = []
    list_time = []
    for model in word_embedding_model:
        mat_distance_wmd = np.array(lecture_fichier_distances_wmd(f"distances_{model}.7z"))
        for dim in range(1,6):
            mds_model = MDS(n_components=dim, dissimilarity="precomputed",n_jobs=-1)
            print(f"{model}:{dim} fit MDS",end="\r")
            t0 = time.time()
            mds_embedding = mds_model.fit_transform(np.array(mat_distance_wmd))
            t1 = time.time() - t0
            mds_distance = euclidean_distances(mds_embedding)
            print(f"{model}:{dim} temps: {t1} calcul correlation",end="\r")
            r_pearson = correlation_epsilon(mat_distance_wmd,mds_distance,epsilon=np.inf,type="pearson")
            r_spearman = correlation_epsilon(mat_distance_wmd,mds_distance,epsilon=np.inf,type="spearman")

            list_model.append(model)
            list_dim.append(dim)
            list_corr_p.append(r_pearson)
            list_corr_s.append(r_spearman)
            list_stress.append(mds_model.stress_)
            list_time.append(t1)

            df_evaluation = pd.DataFrame(list(zip(
                    list_model, list_dim, list_corr_p,list_corr_s, list_stress,list_time)),
                                                columns=[ "model", "dim", "corr_p","corr_s","stress","time"])
            df_evaluation.to_csv("data/tunning/mds_wmd.csv",sep=";",index=False)


