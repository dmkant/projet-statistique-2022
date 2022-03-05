#à tester

from gensim import models
import pandas as pd
cbow_model = models.KeyedVectors.load_word2vec_format("word_embedding/tunning/cbow.kv")
data = pd.DataFrame(kv_model.vectors)
data.index = cbow_model.index_to_key 

from sklearn.manifold import TSNE
#choix du nb de composantes
tsne = TSNE(n_components=2) 
tsne_sortie= tsne.fit_transform(data)
print(tsne_sortie.shape)

