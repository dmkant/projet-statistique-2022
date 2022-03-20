from typing import Union,List

import numpy as np


def TF(doc: List[str], vocabulaire: Union[list, set] = None, type: str = 'frequence_brute', **kwargs) -> dict:

    if vocabulaire is not None:
        vocabulaire = set(vocabulaire).union(set(doc))
    
    else:
        vocabulaire = set(doc)

    frequenceBrute = {v: 0 for v in vocabulaire}

    for mot in doc:
        frequenceBrute[mot] += 1

    frequenceBrute = {v: frequenceBrute[v] / len(doc) for v in vocabulaire}

    if type == 'frequence_brute':
        return frequenceBrute

    elif type == 'binaire':
        return {v: frequenceBrute[v] > 0 for v in vocabulaire}
    
    elif type == 'normalisation':
        K: float = kwargs['K'] if 'K' in kwargs else .5

        maximum:float = max(frequenceBrute.values())
        return {v: K + (1 - K) * frequenceBrute[v] / maximum for v in vocabulaire}

def TF_docs(docs: List[List[str]], vocabulaire: Union[list, set] = None, type: str = 'frequence_brute', **kwargs) -> List[dict]:

    tfDocs = []
    for doc in docs:
       tfDocs.append(TF(doc, vocabulaire, type, **kwargs))
    
    return tfDocs

def IDF(docs: List[List[str]], vocabulaire: Union[list, set] = None):

    vocabulairePresentDocuments = set()
    for doc in docs:
        vocabulairePresentDocuments.update(set(doc))

    if vocabulaire is not None:
        vocabulaire = set(vocabulaire).union(vocabulairePresentDocuments)

    else:
        vocabulaire = vocabulairePresentDocuments

    idf = {v: np.inf for v in vocabulaire}
    frequence = {v: sum(v in doc for doc in docs) for v in vocabulaire}
    
    for mot in vocabulaire:
        if frequence[mot] > 0:
            idf[mot] = np.log(len(docs) / frequence[mot])

    
    return idf

def TF_IDF(docs: List[List[str]], vocabulaire: Union[list, set] = None) -> List[dict]:

    idf = IDF(docs, vocabulaire)
    
    vocabulaire = set(idf.keys())

    retour = []
    for doc in docs:
        tf = TF(doc, vocabulaire)
        tfidf = {v: tf[v] * idf[v] for v in vocabulaire}
        retour.append(tfidf)

    return retour
