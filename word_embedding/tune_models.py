from importlib.resources import path
import pandas as pd
import json
import numpy as np
import scipy as sp
import random
from gensim import models
import os 
from comparaison_BATS import *


def get_df_tag_similarity(read:bool=True,test_size:int = 10):
    if read:
        df_tag_similiraty = pd.read_csv("word_embedding/tunning/tag_similiraty.csv",sep=";",index_col=0)
    else:
        # Get all tags
        # load sentences
        with open("data/docs.json") as file:
            docs = json.load(file)
        # original data
        with open("data/req_makeorg_environnement.json",encoding="utf8") as file:
            dict_req_makeorg = json.load(file)
        list_tags =[[proposition["tags"][j]["label"] for j in range(len(proposition["tags"]))] for proposition in dict_req_makeorg["results"]]


        # on ne garde que les proppsition tague
        docs = [docs[i] for i in range(len(list_tags)) if len(list_tags[i]) > 0]
        list_tags = [list_tags[i] for i in range(len(list_tags)) if len(list_tags[i]) > 0]
        list_lemme = np.unique([docs[i][j] for i in range(len(docs)) for j in range(len(docs[i]))])
        list_tags_unique = np.unique([list_tags[i][j] for i in range(len(list_tags)) for j in range(len(list_tags[i]))])

        word_dictionary = list_lemme #inutile

        # Evaluation on a sample
        dictionary_test = random.sample(list(word_dictionary),test_size)
        mat_tag_distance = np.array([[np.sum([1 if tag in list_tags[i] and word in docs[i]
                                            else 0
                                            for i in range(len(docs))])
                                    for tag in list_tags_unique]
                                    for word in dictionary_test])

        matx_tag_similiraty = np.zeros((mat_tag_distance.shape[0],mat_tag_distance.shape[0]))
        np.fill_diagonal(matx_tag_similiraty,1)
        for i in range(mat_tag_distance.shape[0]):
            print(i, end="\r")
            for j in range(i):
                matx_tag_similiraty[i][j] = 1-sp.spatial.distance.cosine(mat_tag_distance[i],mat_tag_distance[j])
                matx_tag_similiraty[j][i] = matx_tag_similiraty[i][j]

        # Save matx_tag_similarity
        df_tag_similiraty = pd.DataFrame(matx_tag_similiraty,index=dictionary_test,columns=dictionary_test)
        df_tag_similiraty.to_csv("word_embedding/tunning/tag_similiraty.csv",sep=";")
    
    return(df_tag_similiraty)

def tag_evaluation(embed_model,df_tag_similiraty):
    matx_similiraty = np.zeros((df_tag_similiraty.shape[0],df_tag_similiraty.shape[0]))
    np.fill_diagonal(matx_similiraty,1)
    for k in range(df_tag_similiraty.shape[0]):
        for j in range(k):
            matx_similiraty[k][j] = 1-sp.spatial.distance.cosine(embed_model.get_vector(df_tag_similiraty.index[k]),
                                                                embed_model.get_vector(df_tag_similiraty.index[j]))
            matx_similiraty[j][k] = matx_similiraty[k][j]
    
    return(np.mean((matx_similiraty - np.array(df_tag_similiraty))**2)/2)


df_tag_similiraty = get_df_tag_similarity(read=False,test_size=300)
# modeleReference: models.KeyedVectors = models.KeyedVectors.load_word2vec_format("data/w2vec.bin", 
#                                                                        binary=True, unicode_errors="ignore")

#Tuning parameters
list_models_filename = os.listdir("word_embedding/training_models")
list_windows = []
list_dim_emb = []
list_type_model = []
#Evaluation metrics
list_tag_mse = []
list_ref_err_dis_cos = []
list_ref_rmse_dis_cos = []
list_ref_err_moy_freq = []
list_ref_rmse_freq = []

for models_filename in list_models_filename:
    print(models_filename)
    embed_model = models.KeyedVectors.load_word2vec_format(f"word_embedding/training_models/{models_filename}")
    tune_param = models_filename.split("_")
    list_type_model.append(tune_param[0])
    list_windows.append(tune_param[1])
    list_dim_emb.append(tune_param[2].split(".")[0])

    #tag evaluation
    list_tag_mse.append(tag_evaluation(embed_model=embed_model,df_tag_similiraty=df_tag_similiraty))
    #Reference evaluation
    # stats = get_stats_comparaisons_BATS(embed_model, modeleReference)
    # list_ref_err_dis_cos.append(stats["err_dis_cos"])
    # list_ref_rmse_dis_cos.append(stats["rmse_dis_cos"])
    # list_ref_err_moy_freq.append(stats["err_moy_freq"])
    # list_ref_rmse_freq.append(stats["rmse_freq"])

    list_ref_err_dis_cos.append(np.NAN)
    list_ref_rmse_dis_cos.append(np.NAN)
    list_ref_err_moy_freq.append(np.NAN)
    list_ref_rmse_freq.append(np.NAN)


df_evaluation = pd.DataFrame(list(zip(
    list_models_filename, list_type_model, list_windows, list_dim_emb,
    list_tag_mse,list_ref_err_dis_cos,list_ref_rmse_dis_cos,list_ref_err_moy_freq,list_ref_rmse_freq)),
                                 columns=[ "models_filename", "type_model", "windows", "dim_emb",
    "tag_mse","ref_err_dis_cos","ref_rmse_dis_cos","ref_err_moy_freq","ref_rmse_freq"])

df_evaluation.to_csv("word_embedding/tunning/evaluation.csv",sep=";",index=False)