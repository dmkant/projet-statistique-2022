import pandas as pd
import itertools
from glove import Corpus, Glove # creating a corpus object
import time
import json
import numpy as np
import sklearn.metrics.pairwise as metric
import scipy as sp
import random
import matplotlib.pyplot as plt
from gensim import models

def save_glove(glove_model,file_name):
    with open(file_name, "w") as f:
        f.write(str(len(glove_model.dictionary))+" "+str(glove_model.no_components))
        f.write("\n")
        for word in glove_model.dictionary:
            f.write(word)
            f.write(" ")
            for i in range(0, glove_model.no_components):
                f.write(str(glove_model.word_vectors[glove_model.dictionary[word]][i]))
                f.write(" ")
            f.write("\n")
    



# load sentences
with open("data/docs.json") as file:
    docs = json.load(file)
    
    
    
list_windows = [2,3,4,5,6,7,8,9,10,15]
list_dim_emb = [i for i in range(10,210,20)]


for window in list_windows:
    for dim_emb in list_dim_emb:
        print(window,"-",dim_emb)
        corpus = Corpus() 
        corpus.fit(docs, window=window)
        # train glove
        glove_model = Glove(no_components=dim_emb, learning_rate=0.001)
        glove_model.fit(corpus.matrix, epochs=30, no_threads=2)
        glove_model.add_dictionary(corpus.dictionary)
        #save glove_model
        save_glove(glove_model=glove_model,file_name="word_embedding/training_models/glove_"+str(window)+"_"+str(dim_emb)+".kv")
        #Word2vec models
        w2vec_skipgramm = models.Word2Vec(sentences = docs, size = dim_emb, window = window, min_count = 1, workers = 6, sg = 1)
        w2vec_cbow = models.Word2Vec(sentences = docs, size = dim_emb, window = window, min_count = 1, workers = 6, sg = 0)

        #save
        w2vec_skipgramm.wv.save_word2vec_format("word_embedding/training_models/skipgramm_"+str(window)+"_"+str(dim_emb)+".kv")
        w2vec_cbow.wv.save_word2vec_format("word_embedding/training_models/cbow_"+str(window)+"_"+str(dim_emb)+".kv")