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

comparer = input("Comparer avec et sans parrallelisation (y/n) ?: ")

def countdown(n,s=0):
    print(f"debut {s}")
    while n>s:
        n -= 1
    print(f"fin {s}")
    return n

if comparer == "y":
    t = time.time()
    for _ in range(20):
        print(countdown(10**7), end=" ")
    print(f"Sans paralellisation: {time.time() - t}")  
    # takes ~10.5 seconds on medium sized Macbook Pro


    t = time.time()
    results = Parallel(n_jobs=2)(delayed(countdown)(10**7) for _ in range(20))
    print(results)
    print(f"Avec paralellisation n_job=2: {time.time() - t}") 

    t = time.time()
    results = Parallel(n_jobs=20)(delayed(countdown)(10**7, _) for _ in range(20))
    print(results)
    print(f"Avec paralellisation n_job=2: {time.time() - t}")
    
    t = time.time()
    with multiprocessing.Pool() as pool:
        results = [pool.apply_async(countdown, (10**7, _)) for _ in range(20)]
        results = [f.get() for f in results]
    print(f"Avec paralellisation: {time.time() - t}")
    
    
compare_mantel = input("Comparer Mantel test (y/n) ?: ")

if compare_mantel == "y":
    print("Lecture distance")
    #Read WMD matrix 
    mat_distance_wmd = lecture_fichier_distances_wmd("distances_cbow.7z")
    flat_mat_distance_wmd = mat_distance_wmd.values[np.triu_indices(mat_distance_wmd.shape[0])]
    mat_distance_wmd_thr = np.copy(mat_distance_wmd)
    mat_distance_wmd_thr[np.where(mat_distance_wmd_thr >= np.quantile(flat_mat_distance_wmd,0.75))] = np.quantile(flat_mat_distance_wmd,0.75)
    
    # mat_distance_wmd = np.random.randint(-2000,2000,size=(9500,9500))
    # mat_distance_wmd = (mat_distance_wmd + mat_distance_wmd.T)/2
    # np.fill_diagonal(mat_distance_wmd,0)
    #On considere mat_distance_wmd comme le doc embedding et on calcule la matrice de distance euclidiennes
    mat_distance_euclidean = euclidean_distances(mat_distance_wmd)
    mat_distance_manhattan = manhattan_distances(mat_distance_wmd)
    mat_distance_euclidean_thr = euclidean_distances(mat_distance_wmd_thr)
    
    #tsne embedding
    tsne_vec = TSNE(n_components=3,metric="euclidean") 
    tsne_dist = TSNE(n_components=3,metric="precomputed") 
    mat_tsne_vec = tsne_vec.fit_transform(mat_distance_wmd)
    mat_tsne_dist = tsne_vec.fit_transform(mat_distance_wmd)
    
    mat_distance_tsne_vec = euclidean_distances(mat_tsne_vec)
    mat_distance_tsne_dist = euclidean_distances(mat_tsne_dist)
    print("Fin lecture")
    
    # # Mantel module
    # t = time.time()
    # resu = mantel_parallel.test(mat_distance_wmd,
    #             mat_distance_euclidean,
    #             method="spearman",
    #             perms=300)
    # print(f"Temps Mantel module {time.time() - t}")
    # print(f"Resulat : {resu}")
        
    # Parallel version
    t = time.time()
    resu = mantel_parallel.test(mat_distance_wmd,
                                mat_distance_euclidean,
                                method="spearman",
                                perms=300,
                                parallel=True,
                                n_jobs=5)
    print(f"Temps Parallel module {time.time() - t}")
    print(f"Resulat euclidiean: {resu}")
    
    t = time.time()
    resu = mantel_parallel.test(mat_distance_wmd,
                                mat_distance_manhattan,
                                method="spearman",
                                perms=300,
                                parallel=True,
                                n_jobs=5)
    print(f"Temps Parallel module {time.time() - t}")
    print(f"Resulat Manhattan: {resu}")
    

    t = time.time()
    resu = mantel_parallel.test(mat_distance_wmd_thr,
                                mat_distance_euclidean_thr,
                                method="spearman",
                                perms=300,
                                parallel=True,
                                n_jobs=5)
    print(f"Temps Parallel module {time.time() - t}")
    print(f"Resulat Euclidiean threashold: {resu}")
    
    t = time.time()
    resu = mantel_parallel.test(mat_distance_wmd,
                                mat_distance_tsne_vec,
                                method="spearman",
                                perms=300,
                                parallel=True,
                                n_jobs=5)
    print(f"Temps Parallel module {time.time() - t}")
    print(f"Resulat TSNE Vec: {resu}")
    
    t = time.time()
    resu = mantel_parallel.test(mat_distance_wmd,
                                mat_distance_tsne_dist,
                                method="spearman",
                                perms=300,
                                parallel=True,
                                n_jobs=5)
    print(f"Temps Parallel module {time.time() - t}")
    print(f"Resulat TSNE Dist: {resu}")

    t = time.time()
    resu = mantel_parallel.test(mat_distance_wmd_thr,
                                mat_distance_tsne_dist,
                                method="spearman",
                                perms=300,
                                parallel=True,
                                n_jobs=5)
    print(f"Temps Parallel module {time.time() - t}")
    print(f"Resulat TSNE Dist threshold: {resu}")

    