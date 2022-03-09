import sys

import numpy as np
from gensim import models

sys.path.append('..')
from stats_descriptives import tf_idf as TFIDF


def word_emb_vers_doc_emb_moyenne(docs: list[list[str]], modele: models.KeyedVectors, methode: str = 'TF-IDF') -> list[np.ndarray]:

    moyennes = []
    if methode == 'TF-IDF':
        ponderations = TFIDF.TF_IDF(docs)
        # Les pondérations ne valent pas forcément 1 en somme,
        # on normalise
        for i in range(len(docs)):
            somme = sum(ponderations[i].values())
            if somme != 1:
                #[print(ponderations[i][v]) for v in ponderations[i]]
                ponderations[i] = {v: ponderations[i][v] / somme for v in ponderations[i]}

    elif methode == 'TF':
        # La somme des pondérations vaut 1 par définition
        ponderations = TFIDF.TF_docs(docs)

    for i in range(len(docs)):

        representation: np.ndarray = None

        for v in ponderations[i]:

            if ponderations[i][v] != 0:

                if representation is None:
                    representation = ponderations[i][v] * modele[v]

                else:
                    representation += ponderations[i][v] * modele[v]

        moyennes.append(representation)
   
   
    return moyennes
