import numpy as np
from gensim import models


def wmd_docs(docs: list[list[str]], modele: models.KeyedVectors, posDocBase: int, posAutresDocs: int | list[int] = None) -> float | list[float]:
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


def distance_wmd_tous_docs(docs: list[list[str]], modele: models.KeyedVectors, retour = 'fichier', nomFichier = "distances.txt", toInteger = True) -> list[np.array]:
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
        cheminFichier = '../data/' + nomFichier
        fichier = open(cheminFichier, 'w')

    nbTotal: int = len(docs) * (len(docs) - 1) / 2 
    print(f"Nombre de distances à calculer : {nbTotal}")
    print("0 % |", end = ' ')
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
                print(f"{percent} % |", end = " ")
        
        if retour == 'fichier': 

            for v in temp[:-1]:
                fichier.write(str(v) + '\t')
                
            fichier.write(str(temp[-1]))

            if i != len(docs) - 2:
                fichier.write('\n')
    
    if retour == 'fichier':
        fichier.close()
        print("Données enregistrées dans " + cheminFichier)
        return None

    elif retour == 'liste':
        return distancesDocs
    
def lecture_fichier_distances_wmd(chemin: str = "../data/distances.txt", integer = True) -> list[np.ndarray]:
    
    typeData = np.uint32 if integer else np.float32
    distances = []

    fichier = open(chemin, 'r')
    for ligne in fichier.readlines():
        ligne = ligne.split('\t')
        distances.append(np.array([int(v) if integer else float(v) for v in ligne], dtype = typeData))
        del ligne
        
    fichier.close()

    return distances
