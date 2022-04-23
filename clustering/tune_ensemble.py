from gensim import models
import numpy as np
import pandas as pd
import json
from sklearn.manifold import TSNE

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

def init_worker(mat_doc_embedding,mat_doc_embedding_shape,allLabels,allLabels_shape,allSilhouette,type_doc_embedding,type_word_embedding):
    var_dict['mat_doc_embedding'] = mat_doc_embedding
    var_dict['mat_doc_embedding_shape'] = mat_doc_embedding_shape
    var_dict['allLabels'] = allLabels
    var_dict['allLabels_shape'] = allLabels_shape
    var_dict['allSilhouette'] = allSilhouette
    var_dict["type_doc_embedding"] = type_doc_embedding
    var_dict["type_word_embedding"] = type_word_embedding


def Ensemble_evaluation(perplexity,listeK):
    allLabels_np = np.frombuffer(var_dict['allLabels']).reshape(var_dict['allLabels_shape'])
    mat_doc_embedding_np = np.frombuffer(var_dict['mat_doc_embedding']).reshape(var_dict['mat_doc_embedding_shape'])
    
    listSolver = ['hgpa','mcla','hbgf']
    listSolver = ["mcla"]
    listMinSilhouette = [-1,0,0.45]
    if var_dict["type_doc_embedding"] == "WMD_Distance" and perplexity is None:
        df_result,_ = CL.Ensemble_Clustering(distance=mat_doc_embedding_np,allLabels=allLabels_np,allSilhouette=var_dict['allSilhouette'],listeK=listeK,listSolver=listSolver,listMinSilhouette=listMinSilhouette)
    else:
        df_result,_ = CL.Ensemble_Clustering(data=mat_doc_embedding_np,allLabels=allLabels_np,allSilhouette=var_dict['allSilhouette'],listeK=listeK,listSolver=listSolver,listMinSilhouette=listMinSilhouette)
        
    df_result["perplexity"] = perplexity
    df_result["wordEmbedding"] = var_dict["type_word_embedding"]
    docEmbedding,docEmbedding2 = var_dict["type_doc_embedding"].split("_")
    df_result["docEmbedding"] = docEmbedding
    df_result["docEmbedding2"] = docEmbedding2
    df_result["listClustering"] = "kmeans" if var_dict["type_doc_embedding"] == "WMD_Distance"  and perplexity is None else "-".join(["kmeans","gmm"])


    return df_result

def Ensemble_parallel(word_embedding_model,list_perplexity,n_jobs=5):
    with open('data/docs.json', encoding = "utf8") as f:
        docs = json.load(f)

    list_df_evaluation = []
    for model in word_embedding_model:
        print(f"{model} : lecture des matrices...")
        ev = models.KeyedVectors.load_word2vec_format(f"data/tunning/{model}.kv")
        dict_mat_embedding = {}
        #Read moy matrix
        dict_mat_embedding["Moyenne_TF"] = moyenne.word_emb_vers_doc_emb_moyenne(docs, ev, methode = 'TF')
        dict_mat_embedding["Moyenne_TFIDF"]  = moyenne.word_emb_vers_doc_emb_moyenne(docs, ev, methode = 'TF-IDF')
        if model != "glove":
            dict_mat_embedding["WMD_MDS"] = np.array(pd.read_csv(f"data/tunning/MDS/{model}_mds_embedding.csv",sep=";",header=0))
        dict_mat_embedding["WMD_Distance"] = np.array(lecture_fichier_distances_wmd(f"distances_{model}.7z"))

        for type_doc_embedding,mat_doc_embedding in dict_mat_embedding.items():
            print(f"entrainement: {type_doc_embedding}")

            if type_doc_embedding == "WMD_Distance":
                list_perplexity2 = [perplexity for perplexity in list_perplexity if perplexity is not None]
            else:
                list_perplexity2 = [perplexity for perplexity in list_perplexity]
            for perplexity in list_perplexity2:
                ensembleK = range(2,15)
                
                if perplexity is not None:
                    metric = "precomputed" if type_doc_embedding == "WMD_Distance" else "euclidean"
                    tsne = TSNE(n_components = 2, perplexity=perplexity, n_iter=2000, random_state=0,metric=metric)
                    tsne_moy = tsne.fit_transform(mat_doc_embedding)
                    allLabels,allSilhouette = CL.Get_All_Labels(data=tsne_moy,ensembleK=ensembleK,listAlgo= ["kmeans","gmm"]) 
                    
                    mat_doc_embedding_shared = multiprocessing.RawArray('d', tsne_moy.shape[0]*tsne_moy.shape[1])
                    mat_doc_embedding_np = np.frombuffer(mat_doc_embedding_shared).reshape(tsne_moy.shape)
                    np.copyto(mat_doc_embedding_np, tsne_moy)                  
                else:
                    if type_doc_embedding == "WMD_Distance":
                        allLabels,allSilhouette = CL.Get_All_Labels(distance=mat_doc_embedding,ensembleK=ensembleK,listAlgo= ["kmeans"])
                    else:
                        allLabels,allSilhouette = CL.Get_All_Labels(data=mat_doc_embedding,ensembleK=ensembleK,listAlgo= ["kmeans","gmm"])
                    
                    mat_doc_embedding_shared = multiprocessing.RawArray('d', mat_doc_embedding.shape[0]*mat_doc_embedding.shape[1])
                    mat_doc_embedding_np = np.frombuffer(mat_doc_embedding_shared).reshape(mat_doc_embedding.shape)
                    np.copyto(mat_doc_embedding_np, mat_doc_embedding)   


                allLabels_shared = multiprocessing.RawArray('d', allLabels.shape[0]*allLabels.shape[1])
                allLabels_np = np.frombuffer(allLabels_shared).reshape(allLabels.shape)
                np.copyto(allLabels_np, allLabels)        
                
                listeK = range(2,15)
                with multiprocessing.Pool(processes=n_jobs,initializer=init_worker, initargs=(mat_doc_embedding_shared,mat_doc_embedding_np.shape,allLabels_shared,allLabels.shape,allSilhouette,type_doc_embedding,model)) as pool:
                    results = [pool.apply_async(Ensemble_evaluation,(perplexity,[K])) for K in listeK]
                    df_evaluation = pd.concat([f.get() for f in results])
                    df_evaluation.to_csv(f"data/tunning/clustering/Ensemble/ensemble_{model}_{type_doc_embedding}.csv",sep=";",index=False)

                    list_df_evaluation += [df_evaluation]
                    pd.concat(list_df_evaluation).to_csv(f"data/tunning/clustering/ensemble.csv",sep=";",index=False)


use_parrallel = True
word_embedding_model = ["cbow","glove","skipgram"]
n_jobs = 25
list_perplexity = [None,200]

if use_parrallel:
    Ensemble_parallel(word_embedding_model,list_perplexity,n_jobs=n_jobs)    
else:
    pass