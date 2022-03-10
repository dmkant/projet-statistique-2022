import os

import numpy as np
import py7zr
from gensim import models
from pandas import DataFrame


def wmd_docs(docs: list[list[str]], modele: models.KeyedVectors, posDocBase: int, posAutresDocs: list[int] = None) -> list[float]:
    """
    Renvoie les distances entre un document et un ensemble d'autres documents

    Args:
        docs (list[list[str]]): Liste de documents
        modele (models.KeyedVectors): Modèle word-embedding gensim
        posDocBase (int): Position du document de référence da,s 'docs'
        posAutresDocs (int | list[int], Defaults None): Position du ou des autres documents dans la liste 'docs' dont
         doit être mesurée la distance avec le document de référence. Si vaut 'None' la distance est calculée pour tous
         les documents du corpus

    Returns:
        float | list[float]: Les distances wmd entre le document de référence et ceux associés aux positions
        de posAutresDocs. S'il n'y a qu'un seul document, la valeur rendue est un flottant (pas une liste de flottants).
    """
    if posAutresDocs is None:
        posAutresDocs = range(len(docs))

    elif isinstance(posAutresDocs, int):
        posAutresDocs = [posAutresDocs]
    
    distances: list[float] = []
    docBase = docs[posDocBase]
    for i in posAutresDocs:

        if i == posDocBase:
            distances.append(0.)

        else:
            """
            When using this code, please consider citing the following papers:
            Ofir Pele and Michael Werman “A linear time histogram metric for improved SIFT matching”
            Ofir Pele and Michael Werman “Fast and robust earth mover’s distances”
            Matt Kusner et al. “From Word Embeddings To Document Distances”.
            (issu de la page KeyedVectors, gensim), à propos de wmdistance
            """
            distances.append(modele.wmdistance(docBase, docs[i]))
    
    return distances[0] if len(distances) == 1 else distances


def distance_wmd_tous_docs(docs: list[list[str]], modele: models.KeyedVectors, retour = 'fichier', nomFichier = "distances.7z", toInteger = True) -> list[np.array]:
    """
    ATTENTION : Peut être très très LONG ! il y a n * (n - 1) / 2 distances à estimer (n la taille du corpus)
    Calcule la distance wmd sur l'entièreté des documents. Calcule pour tout indice
    (i, j) tel que 0 < i < j < nombre de documents.
    POur l'enregistrement du fichier, compter 4 octets dans le mode entier par distance. Ce qui fait
    2 * n * (n - 1) octets par corpus.
    """
    typeStockage = np.uint32 if toInteger else np.float32
    if retour == "liste":
        distancesDocs = [np.array([0.] * (len(docs) - 1 - i), dtype = typeStockage) for i in range(len(docs) - 1)]
    
    elif retour == 'fichier':
        path = "data/distances/"
        cheminFichier = path + 'wmd.txt'
        fichier = open(cheminFichier, 'w')
        fichier.write(('integer' if toInteger else 'float') + '\n')
        

    nbTotal: int = len(docs) * (len(docs) - 1) / 2 
    print(f"Nombre de distances à calculer : {nbTotal}")
    print("0 % |",end=" ",flush=True)
    nbFaits: int = 0
    for i, doc in enumerate(docs[:-1]):

        if retour == 'fichier':
            temp = np.zeros(len(docs) - 1 - i, dtype = typeStockage)
            
        for j in range(i + 1, len(docs)):
            """
            When using this code, please consider citing the following papers:
            Ofir Pele and Michael Werman “A linear time histogram metric for improved SIFT matching”
            Ofir Pele and Michael Werman “Fast and robust earth mover’s distances”
            Matt Kusner et al. “From Word Embeddings To Document Distances”.
            (issu de la page KeyedVectors, gensim), à propos de wmdistance
            """

            d: float = modele.wmdistance(doc, docs[j])
            if toInteger:
                d: int = round(1000 * d)
            if retour == 'liste':
                distancesDocs[i][j-(i+1)] = d
            
            elif retour == 'fichier':
                temp[j-(i+1)] = d

            nbFaits += 1
            if (percent := round(nbFaits * 100 / nbTotal)) % 5 == 0 and round((nbFaits - 1) * 100 / nbTotal) % 5 != 0:
                print(f"{percent} % |", end = " ",flush=True)
        
        if retour == 'fichier': 

            for v in temp[:-1]:
                fichier.write(str(v) + '\t')
                
            fichier.write(str(temp[-1]))

            if i != len(docs) - 2:
                fichier.write('\n')
    
    if retour == 'fichier':
        fichier.close()
        print("Calculs terminés. Enregistrement dans une archive.")
        cheminArchive = path + nomFichier
        archive = py7zr.SevenZipFile(cheminArchive, 'w')
        archive.write(cheminFichier, arcname = 'wmd.txt')
        archive.close()
        os.remove(cheminFichier)
        print("Données enregistrées dans " + cheminArchive)

        return None

    elif retour == 'liste':
        return distancesDocs
    
def lecture_fichier_distances_wmd(nomFichier: str = "distances.7z") -> list[np.ndarray]:
    
    estUneArchive: bool = False
    path: str = "data/distances/"
    cheminFichier = path + nomFichier
    if cheminFichier[-3:] == '.7z':
        estUneArchive = True
        with py7zr.SevenZipFile(cheminFichier, 'r') as archive:
            archive.extractall(path = path)
        cheminFichier = path + 'wmd.txt'
    
    distances = []

    fichier = open(cheminFichier, 'r')
    lignesFichier = fichier.readlines()

    # Suppression des '\n' en fin de ligne
    lignesFichier = [ligne.replace('\n', '') for ligne in lignesFichier]

    # Gestion du type de données
    enTete: str = lignesFichier[0]
    typeData = np.uint32 if enTete == 'integer' else np.float32
    lignesFichier: list[str] = lignesFichier[1:]

    # Séparation des données numériques par séparateur '\t'
    lignesFichier: list[list[str]] = [ligne.split('\t') for ligne in lignesFichier]

    # Nombre de documents étudiés
    N = len(lignesFichier) + 1
    distances = np.zeros((N,N), dtype = typeData)
    for i in range(N - 1):
        for j in range(i + 1, N):
            distances[i,j] = distances[j,i] = typeData(lignesFichier[i][j - (i + 1)])
        
    fichier.close()

    if estUneArchive:
        os.remove(cheminFichier)


    return DataFrame(distances)


if __name__ == "__main__":
    import json
    import time
    
    # load sentences
    with open("data/docs.json") as file:
        docs = json.load(file)
    for model_type in ["skipgram","glove"]:
        print(model_type)
        t0 = time.time()
        embed_model = models.KeyedVectors.load_word2vec_format(f"data/tunning/{model_type}.kv")
        distance_wmd_tous_docs(docs,
                                modele = embed_model, 
                                retour = 'fichier', 
                                nomFichier = f"distances_{model_type}.7z",
                                toInteger = True)
        print(f"\n Temps de calcul: {time.time()-t0}/3600")