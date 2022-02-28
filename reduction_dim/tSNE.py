#à tester

data

from sklearn.manifold import TSNE
#choix du nb de composantes
tsne = TSNE(n_components=2) 
tsne_sortie= tsne.fit_transform(data)
print(tsne_sortie.shape)

