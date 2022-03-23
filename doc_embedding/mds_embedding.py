from operator import mod
from sklearn.manifold import MDS
import pandas as pd
import time



from sklearn.metrics.pairwise import euclidean_distances
import scipy.spatial as sp
import scipy.stats
import random
import math

from word_embedding.distance_wmd import *
from reduction_dim.correlation_matrix import *

word_embedding_model = ["cbow","skipgram","glove"]

list_model = []
list_dim = []
list_corr_p = []
list_corr_s = []
list_stress = []
list_time = []

for model in word_embedding_model:
    mat_distance_wmd = lecture_fichier_distances_wmd(f"distances_{model}.7z")
    for dim in range(1,6):
        mds_model = MDS(n_components=dim, dissimilarity="precomputed")
        print(f"{model}:{dim} fit MDS",end="\r")
        t0 = time.time()
        mds_embedding = mds_model.fit_transform(mat_distance_wmd)
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


