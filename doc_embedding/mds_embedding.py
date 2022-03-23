

from gensim import models
from sklearn.manifold import TSNE, MDS
import pandas as pd
import time


from sklearn.metrics.pairwise import euclidean_distances
import scipy.spatial as sp
import scipy.stats
import random
import math

from stats_descriptives import tf_idf as TFIDF

from word_embedding.distance_wmd import *
from reduction_dim.correlation_matrix import *


mat_distance_wmd = lecture_fichier_distances_wmd("distances_cbow.7z")
mds_model = MDS(n_components=3, dissimilarity="precomputed")
mds_embedding = mds_model.fit_transform(mat_distance_wmd)
mds_distance = euclidean_distances(mds_embedding)

r_pearson = correlation_epsilon(mat_distance_wmd,mds_distance,epsilon=np.inf)
