#fichier sous la forme d'un dataframe
from gensim import models
import pandas as pd
cbow_model = models.KeyedVectors.load_word2vec_format("word_embedding/tunning/cbow.kv")
data = pd.DataFrame(cbow_model.vectors)
data.index = cbow_model.index_to_key 

import numpy as np
from sklearn import decomposition
from sklearn import preprocessing
import matplotlib.pyplot as plt

X = data.values

# Centrage et Réduction
std_scale = preprocessing.StandardScaler().fit(X)
X_scaled = std_scale.transform(X)

# Calcul des composantes principales
pca = decomposition.PCA(n_components=6)
pca.fit(X_scaled)

#proportion de variance expliquée
print(pca.explained_variance_ratio_)

#nombre d'observations
n = data.shape[0]

#variance expliquée
eigval = (n-1)/n*pca.explained_variance_

#scree plot
plt.plot(numpy.arange(1,p+1),eigval)
plt.title("Scree plot")
plt.ylabel("Eigen values")
plt.xlabel("Factor number")
plt.show()