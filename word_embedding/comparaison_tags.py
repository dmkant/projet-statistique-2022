from importlib.resources import path
import pandas as pd
import json
import numpy as np
import scipy as sp
import random
from gensim import models
import os 
from comparaison_BATS import *
import sys



def get_df_tag_similarity(read:bool=True,test_size:int = 10) -> pd.DataFrame:
    """Retourne la matrice de similarite selon le tag embedding

    Args:
        read (bool, optional): Lire ou recalculer la matrice de similarite. Defaults to True.
        test_size (int, optional): Taille de l'echantillon test. Defaults to 10.

    Returns:
        pd.DataFrame: Matrice similarite des lemmes selon le tag embedding
    """
    

    if read:
        df_tag_similiraty = pd.read_csv("data/tunning/tag_similiraty.csv",sep=";",index_col=0)
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
        df_tag_similiraty.to_csv("data/tunning/tag_similiraty.csv",sep=";")
    
    return(df_tag_similiraty)

def tag_evaluation(embed_model:models.KeyedVectors, df_tag_similiraty:pd.DataFrame) -> float:
    """Compute MSE between similarity from embed_model and tag embedding

    Args:
        embed_model (models.KeyedVectors): model to test
        df_tag_similiraty (pd.DataFrame): matrice de similarite selon le tag embedding

    Returns:
        float: MSE
    """
    matx_similiraty = np.zeros((df_tag_similiraty.shape[0],df_tag_similiraty.shape[0]))
    np.fill_diagonal(matx_similiraty,1)
    for k in range(df_tag_similiraty.shape[0]):
        for j in range(k):
            matx_similiraty[k][j] = 1-sp.spatial.distance.cosine(embed_model.get_vector(df_tag_similiraty.index[k]),
                                                                embed_model.get_vector(df_tag_similiraty.index[j]))
            matx_similiraty[j][k] = matx_similiraty[k][j]
    
    return(np.mean((matx_similiraty - np.array(df_tag_similiraty))**2)/2)



def evaluation_tag(df_tag_similiraty:pd.DataFrame, start:int=0) -> None:
    """Evalue all model in data/training_models folder with tag_evaluation metric

    Args:
        df_tag_similiraty (pd.DataFrame): Matrice de similarite selon le tag embedding
        start (int, optional): Index de model de debut. Defaults to 0.
    """
    list_models_filename = os.listdir("data/training_models")
    
    if start != 0:
        df_old_evaluation = pd.read_csv("data/tunning/evaluation_tags.csv",sep=";")
        list_windows = df_old_evaluation["windows"].values
        list_dim_emb = df_old_evaluation["dim_emb"].values
        list_type_model = df_old_evaluation["type_model"].values
        #Evaluation metrics
        list_tag_mse = df_old_evaluation["tag_mse"].values
    else:
        list_windows = []
        list_dim_emb = []
        list_type_model = []
        #Evaluation metrics
        list_tag_mse = []

    for models_filename in list_models_filename[start:]:
        embed_model = models.KeyedVectors.load_word2vec_format(f"data/training_models/{models_filename}")
        tune_param = models_filename.split("_")
        list_type_model.append(tune_param[0])
        list_windows.append(tune_param[1])
        list_dim_emb.append(tune_param[2].split(".")[0])

        #tag evaluation
        print(models_filename,": ",list_models_filename.index(models_filename), " : TAGS",end="\r")    
        list_tag_mse.append(tag_evaluation(embed_model=embed_model,df_tag_similiraty=df_tag_similiraty))
        
        if list_models_filename.index(models_filename) % 5 == 0:
            df_evaluation = pd.DataFrame(list(zip(
                list_models_filename, list_type_model, list_windows, list_dim_emb,list_tag_mse)),
                                            columns=[ "models_filename", "type_model", "windows", "dim_emb","tag_mse"])
            df_evaluation.to_csv("data/tunning/evaluation_tags.csv",sep=";",index=False)
        

if __name__ == "__main__":
    # Bats evaluation
    # evaluation_BATS()
    
    # Tags evaluation
    df_tag_similiraty = get_df_tag_similarity(read=False,test_size=3000)
    evaluation_tag(df_tag_similiraty, start=0)