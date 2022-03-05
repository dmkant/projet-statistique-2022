#pas encore testé avec nos données

#fichier sous la forme d'un dataframe
from gensim import models
import pandas as pd
cbow_model = models.KeyedVectors.load_word2vec_format("word_embedding/tunning/cbow.kv")
data = pd.DataFrame(cbow_model.vectors)
data.index = cbow_model.index_to_key 

#ACP
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

#preparation des données
data = scale(data)

#on choisit le nb de composantes finales
acp = PCA(n_components=3)

acp.fit(data)
data_sortie= acp.fit_transform(X)

y = list(acp.explained_variance_ratio_)
biplot(acp,x=data,cat=y,components=[0,1])
plt.show()

