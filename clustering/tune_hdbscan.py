from gensim import models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from sklearn.manifold import TSNE

import seaborn as sns

from word_embedding.distance_wmd import *
import doc_embedding.moyenne as moyenne
from reduction_dim.correlation_matrix import *
import clustering.fit_clustering as CL 

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import multiprocessing


# A global dictionary storing the variables passed from the initializer.
var_dict = {}

def init_worker(mat_doc_embedding,mat_doc_embedding_shape,type_doc_embedding,type_word_embedding):
    var_dict['mat_doc_embedding'] = mat_doc_embedding
    var_dict['mat_doc_embedding_shape'] = mat_doc_embedding_shape
    var_dict["type_doc_embedding"] = type_doc_embedding
    var_dict["type_word_embedding"] = type_word_embedding


def HDBSCAN_evaluation(perplexity,init_dim):
    mat_doc_embedding_np = np.frombuffer(var_dict['mat_doc_embedding']).reshape(var_dict['mat_doc_embedding_shape'])
    
    metric = "precomputed" if var_dict["type_doc_embedding"] == "WMD_Distance" else "euclidean"
    listeMinClusterSize=[i for i in range(10,15)] + [i for i in range(100,501,50)] + [950]
    listeVoisinage=[i for i in range(2,50,1)] + [100,150,200]
    
    if perplexity is not None:
        tsne = TSNE(n_components = 2, perplexity=perplexity, n_iter=2000, random_state=0,metric=metric)
        tsne_moy = tsne.fit_transform(mat_doc_embedding_np)
        df_result,_ = CL.selection_meilleur_hdbscan(data=tsne_moy,listeDistances="euclidean",init_dim=init_dim, listeMinClusterSize=listeMinClusterSize,listeVoisinage=listeVoisinage)
    else:
        df_result,_ = CL.selection_meilleur_hdbscan(data=mat_doc_embedding_np,listeDistances=metric,init_dim=init_dim,listeMinClusterSize=listeMinClusterSize,listeVoisinage=listeVoisinage)

    df_result["perplexity"] = perplexity
    df_result["wordEmbedding"] = var_dict["type_word_embedding"]
    docEmbedding,docEmbedding2 = var_dict["type_doc_embedding"].split("_")
    df_result["docEmbedding"] = docEmbedding
    df_result["docEmbedding2"] = docEmbedding2

    return df_result

def HDBSCAN_parallel(word_embedding_model,list_perplexity,n_jobs=5):
    with open('data/docs.json', encoding = "utf8") as f:
        docs = json.load(f)

    list_df_evaluation = []
    for model in word_embedding_model:
        print(f"{model} : lecture des matrices...")
        ev = models.KeyedVectors.load_word2vec_format(f"data/tuning/{model}.kv")
        dict_mat_embedding = {}
        #Read moy matrix
        dict_mat_embedding["WMD_Distance"] = np.array(lecture_fichier_distances_wmd(f"distances_{model}.7z"))
        if model != "glove":
            dict_mat_embedding["WMD_MDS"] = np.array(pd.read_csv(f"data/tuning/MDS/{model}_mds_embedding.csv",sep=";",header=0))
        dict_mat_embedding["Moyenne_TF"] = moyenne.word_emb_vers_doc_emb_moyenne(docs, ev, methode = 'TF')
        dict_mat_embedding["Moyenne_TFIDF"]  = moyenne.word_emb_vers_doc_emb_moyenne(docs, ev, methode = 'TF-IDF')
        init_dim = dict_mat_embedding["WMD_MDS"].shape[1] if model != "glove" else dict_mat_embedding["Moyenne_TF"].shape[1]

        for type_doc_embedding,mat_doc_embedding in dict_mat_embedding.items():
            print(f"entrainement: {type_doc_embedding}")

            mat_doc_embedding_shared = multiprocessing.RawArray('d', mat_doc_embedding.shape[0]*mat_doc_embedding.shape[1])
            mat_doc_embedding_np = np.frombuffer(mat_doc_embedding_shared).reshape(mat_doc_embedding.shape)
            np.copyto(mat_doc_embedding_np, mat_doc_embedding)        
            
            with multiprocessing.Pool(processes=n_jobs,initializer=init_worker, initargs=(mat_doc_embedding_shared,mat_doc_embedding.shape,type_doc_embedding,model)) as pool:
                results = [pool.apply_async(HDBSCAN_evaluation,(perplexity,init_dim)) for perplexity in list_perplexity ]
                df_evaluation = pd.concat([f.get() for f in results])
                df_evaluation.to_csv(f"data/tuning/clustering/HDBSCAN/hdbscan_{model}_{type_doc_embedding}.csv",sep=";",index=False)

                list_df_evaluation += [df_evaluation]
                pd.concat(list_df_evaluation).to_csv(f"data/tuning/clustering/hdbscan.csv",sep=";",index=False)


use_parrallel = True
word_embedding_model = ["cbow","glove","skipgram"]
n_jobs = 25
list_perplexity = [None,50,75,100,125,150,200]

if use_parrallel:
    HDBSCAN_parallel(word_embedding_model,list_perplexity,n_jobs=n_jobs)    
else:
    pass