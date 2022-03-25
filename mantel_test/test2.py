import time
from joblib import Parallel, delayed
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
from sklearn.manifold import TSNE,MDS
import sys
import os
sys.path.append(os.getcwd())

import mantel
from word_embedding.distance_wmd import *
import mantel_test.correlation as mantel_parallel


import multiprocessing
from multiprocessing import Process

    
    

print("Lecture distance")
#Read WMD matrix 
mat_distance_wmd = lecture_fichier_distances_wmd("distances_cbow.7z")
mat_distance_euclidean = euclidean_distances(mat_distance_wmd)

print("test")
t = time.time()
resu = mantel_parallel.test(mat_distance_wmd,
                            mat_distance_euclidean,
                            method="spearman",
                            perms=300,
                            parallel=True,
                            n_jobs=5,
                            min_corr = 0.9)
print(f"Temps Parallel module {time.time() - t}")
print(f"Resulat euclidiean: {resu}")
    

    