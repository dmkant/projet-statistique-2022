from time import time

import numpy as np
import pandas as pd
from gensim import models
from numpy.random import randint, seed


# Version basée sur l'algorithme de Dijkstra pour minimiser les calculs de distance
def k_medoides_wmd(docs: list[list[str]], modele: models.KeyedVectors, K: int = 5, iterMax: int = 100, graine: int = None,
percMinDocsGroupe: int = 5, verbose: bool = True, distancesDocs: list[np.ndarray] = None, nbCycles: int = 1, kMeansPlusPlus: bool = True):
    """Applique l'algorithme des k-moyennes à un corpus via un certain modèle de word embedding en utilisant la distance Word Mover's Distance.

    Args:
        docs (list[list[str]]): Le corpus de documents
        modele (models.KeyedVectors): Espace vectoriel de word embedding généré sous gensim
        K (int, optional): Nombre de groupes K à générer. 5 par défaut.
        iterMax (int, optional): Nombre maximal d'itérations par cycle. Défaut à 100.
        graine (int, optional): Graine aléatoire. Si None (défaut) aucune graine n'est définie.
        percMinDocsGroupe (int, optional): Pourcentage de documents que l'on doit retrouver au minimum
         dans chaque groupe. Défaut à 5 (%). Ne doit pas dépasser 100 / K. 
        verbose (bool, optional): Indique si des détails de la progression doivent être donnés. Oui (True) par défaut.
        distancesDocs (list[np.ndarray], optional): Distance entre les documents. Peut ne pas être fournie, auquel cas seules les distances nécessaires
        seront calculées. Peut contenir toutes les distances calculées ou seulement une partie. Les distances non-calculées doivent valoir np.inf.
         L'indice 0 représente les distances du document 0 aux documents 1, 2, ..., N - 1. L'indice 1 représente les distances du document 1
         aux documents 2, 3, ..., N - 1. Longueur de la liste : N - 2. Si le type est entier, sera considéré que la valeur donnée est la valeur arrondie 
         à l'entier le plus proche de 100 fois la distance wmd. None (défaut) indique que les valeurs sont à calculer.
        nbCycles (int, optional): Nombre de cycles qui doivent être réalisés. Est conservé celui qui minimise la variance intra-groupe moyenne. 1 par défaut.
        kMeansPlusPlus (bool, optional): Précise si l'initialisation doit être faite sous l'algorithme kmeans++ (True) ou si elle doit être faite
         de manière totalement aléatoire. Defaults to True. Kmeans++ (True) par défaut.

    Raises:
        Exception: _description_

    Returns:
        _type_: _description_
    """

    if graine is not None:
        seed(graine)
        

    # Stockage des distances : on ne stocke pas les distances i -> i (valeur nulle) ni les
    # i -> j ou i < j (car redondance avec la distance d(j,i))
    # Aussi pour avoir l'accès à la distance entre i et j on prend d(i,j) si i > j et
    # d(j,i) sinon. Supposant le premier cas on devra faire distancesDocs[i, j - (i + 1)]
    # (Suppression des distances j < i et j = i)
    typeInt = np.uint32
    typeFloat = np.float32
    typeData = None
    if distancesDocs is None:
        distancesDocs = [np.array([np.inf] * (len(docs) - 1 - i)) for i in range(len(docs))]
        typeData = typeFloat
    else:
        typeData = distancesDocs[0].dtype

    groupeDocuments = pd.DataFrame(columns = range(len(docs)), dtype = np.uint16)
    positionCentres = pd.DataFrame(columns = range(K))
    nbIter: int = 0
    nbDistancesCalculees: int = 0

    # Variances intragroupe obtenue dans le meilleur cas
    intraVariances = np.array([np.inf] * K)

    # Cas où il y a plusieurs cycles
    if nbCycles > 1:
        
        for i in range(nbCycles):
            
            if verbose:
                print(f"Cycle {i+1} :")

            if graine is not None:
                tempGroupeDocuments, tempPositionCentres, tempNbIter, TempNbDistancesCalculees, tempIntraVariances = k_medoides_wmd(docs = docs, modele = modele, K = K, iterMax = iterMax, graine = graine + 1 + i,
                percMinDocsGroupe = percMinDocsGroupe, verbose = verbose, distancesDocs = distancesDocs, nbCycles = 1)
            
            else:
                tempGroupeDocuments, tempPositionCentres, tempNbIter, TempNbDistancesCalculees, tempIntraVariances = k_medoides_wmd(docs = docs, modele = modele, K = K, iterMax = iterMax,
                percMinDocsGroupe = percMinDocsGroupe, verbose = verbose, distancesDocs = distancesDocs, nbCycles = 1)

                
            # Cas où la variance interne est meilleure sur ce cycle
            if sum(tempIntraVariances) < sum(intraVariances):
                if verbose and i != 0:
                    print("- Meilleure variance intragroupe trouvée")
                    
                intraVariances = tempIntraVariances
                del groupeDocuments
                groupeDocuments = tempGroupeDocuments.iloc[-1, :].copy()
                del tempGroupeDocuments
                del positionCentres
                positionCentres = tempPositionCentres.iloc[-1,:]
            
            nbIter += tempNbIter
            nbDistancesCalculees += TempNbDistancesCalculees

        
        return groupeDocuments, positionCentres, nbIter, nbDistancesCalculees, intraVariances 


    def get_distance(i: int, j: int, /, valeurBrute: bool = False) -> typeData:
        """
        Renvoie la distance WMD entre le document i et le document j. Si cette distance
        est déjà calculée, la renvoie telle quelle. Si elle n'a pas encore été déterminée
        (c'est-à-dire qu'elle vaut + infini) alors elle est calculée.
        
        Args:
            i (int): Indice de l'un des deux documents dont la distance doit être renvoyée
            j (int): Indice du second des deux documents dont la distance doit être renvoyée
            valeurBrute (bool): Indique si la valeur doit être renvoyée de manière brute. C'est-à-dire
             que si elle a déjà été calculée, on renvoie la valeur calculée, sinon on renvoie la valeur
              + infini. Non (False) par défaut.

        Returns:
            uint32 | float32: Valeur de la distance. Le type est fonction des données en entrée de
            la fonction k-means (si oui ou non les distances sont déjà fournies, et si oui sous quel type).
        """
        # valeur brute : vrai si on revoit la valeur même si elle vaut l'infini

        # Cas où les documents sont strictement les mêmes
        if i == j:
            return 0.
        
        else:
            i, j = min(i, j), max(i,j)
            jPrim = j - (i + 1)
            # Cas où la distance n'a pas encore été calculée
            try:

                if not valeurBrute and distancesDocs[i][jPrim] == np.inf:
                    """
                    When using this code, please consider citing the following papers:
                    Ofir Pele and Michael Werman “A linear time histogram metric for improved SIFT matching”
                    Ofir Pele and Michael Werman “Fast and robust earth mover’s distances”
                    Matt Kusner et al. “From Word Embeddings To Document Distances”.
                    (issu de la page KeyedVectors, gensim), à propos de wmdistance
                    """
                
                    if typeData == typeInt:
                        distancesDocs[i][jPrim] = round(100 * modele.wmdistance(docs[i], docs[j]))

                    elif typeData == typeFloat:
                        distancesDocs[i][jPrim] = modele.wmdistance(docs[i], docs[j])


                return distancesDocs[i][jPrim]

            except:
                print(i, j, jPrim, distancesDocs[i][-10:])
                raise Exception(f"Erreur dans l'accès aux distances : i = {i}, j = {j}, jPrim = {jPrim}, Taille de la ligne : {len(distancesDocs[i])}, 10 dernières valeurs de la ligne :{distancesDocs[i][-10:]}")
            
    
    
    
    def get_distances(i: int, indicesDocs: list[int] = range(len(docs)), valeurBrute: bool = False) -> list[typeData]:
        """Renvoie les distances WMD entre le document i et tous les documents dont l'indice est dans indiceDocs

        Args:
            i (int): Indice du document dont la distance à tous les autres doit être renvoyée
            indicesDocs (list[int], optional): Ensemble des documents dont on doit calculer la distance WMD au document i.
            Par défaut : tous les documents du corpus (range(len(docs))).
            valeurBrute (bool, optional): Indique si les valeurs doivent être renvoyées de manière brute. C'est-à-dire
             que si elles ont déjà été calculées, on renvoie les valeurs calculées, sinon on renvoie la valeur
              + infini. Non (False) par défaut.

        Returns:
            list[uint32 | float32]: Liste des distances. Le type est fonction des données en entrée de
            la fonction k-means (si oui ou non les distances sont déjà fournies, et si oui sous quel type).
        """
        distances = np.array([np.inf] * len(indicesDocs), dtype = np.float32)
        for p, j in np.ndenumerate(indicesDocs):
            distances[p] = get_distance(i, j, valeurBrute)

        return distances

    if kMeansPlusPlus:
        
        # Attribution de K centres par méthode k-means++
        "https://www.wikiwand.com/fr/K-moyennes#/Initialisation"

        premiereLigne = np.array([randint(0, len(docs))] + [-1] * (K - 1), dtype = int)
        positionCentres.loc[len(positionCentres.index), :] = premiereLigne
        positionCentres = positionCentres.astype(np.int32)
        d = get_distances(positionCentres.iloc[-1, 0], range(len(docs)))

        for i in range(1, K):
            positionCentres.iloc[-1, i] = np.random.choice(range(len(docs)), p = d / np.sum(d))
            if i != K - 1:
                d = np.minimum(d, get_distances(positionCentres.iloc[-1, i], range(len(docs))))

    else:

        # Attribution de K centres par assignation aléatoires
        positionCentres.loc[len(positionCentres.index), :] = np.random.randint(0, len(docs), size = K)
        

    while (len(groupeDocuments) < 2 or np.any(groupeDocuments.iloc[-1] != groupeDocuments.iloc[-2])) and nbIter < iterMax:

        nbIter += 1
        if verbose:
            print(f"Itération {nbIter} :", end = ' ')
            print("Création des groupes", end = ' ')
            t = time()
        
        groupeDocuments.loc[len(groupeDocuments.index)] = np.zeros(len(docs), dtype = int)
        for centre in positionCentres.iloc[-1]:
            get_distances(centre)
        
        # On associe chaque document à son groupe le plus proche
        for i in range(len(docs)):
            groupeDocuments.iloc[-1,i] = np.argmin([get_distance(i, centre) for centre in positionCentres.iloc[-1]])

        # Cas où il n'y a pas assez de document dans chaque groupe
        pasToucher = set()
        if verbose and 100 * np.min(np.unique(groupeDocuments.iloc[-1], return_counts=True)[1]) / len(docs) < percMinDocsGroupe:
            print("- Correction des tailles de groupe", end = ' ')
        
        # Tant que la part minimale n'est pas atteinte, on pioche dans un groupe qui est de taille suffisante et on
        # l'attribue au groupe de taille minimale.
        # Ne peuvent être piochés les points qui ont déjà changé de groupe dans cette itération, et les groupes qui
        # ont reçu au moins un élément dans cette itération.
        while 100 * np.min(np.unique(groupeDocuments.iloc[-1], return_counts=True)[1]) / len(docs) < percMinDocsGroupe:
            
            index, counts = np.unique(groupeDocuments.iloc[-1], return_counts=True)
            k = index[np.argmin(counts)]
            pasToucher.update(set((np.arange(len(docs)))[groupeDocuments.iloc[-1] == k]))

            ensembleInteret = set(range(len(docs))).difference(pasToucher)
            distances = {i: get_distance(positionCentres.iloc[-1][k], i) for i in ensembleInteret}
            min_val = min(distances.values())
            argmin = [i for i in ensembleInteret if distances[i] == min_val][0]
            pasToucher.add(argmin)
            groupeDocuments.iloc[-1, argmin] = k
                
            
        if verbose:
            print("- part dans chaque groupe (%) : ", end = '')
            for i in range(K):
                print(round(np.sum(groupeDocuments.iloc[-1] == i) * 100 / len(docs)), end = ' ')
                
            print(f"({round(time() - t)} s)", end = ' | ')


        # Cas où il y a stabilité de la position des documents
        if len(groupeDocuments) >= 2 and np.all(groupeDocuments.iloc[-1] == groupeDocuments.iloc[-2]):
            if verbose:
                print("Stabilité atteinte", end = '')
            break
        
        # S'il n'y a pas stabilité on cherche pour chaque catégorie le document le plus proche des autres (wmd) en moyenne
        else:
            if verbose:
                print("Recherche du point moyen pour chaque groupe ( ", end = '')
                t = time()

            positionCentres.loc[len(positionCentres.index)] = (randint(0, len(docs), K))

            for numeroGroupe in range(K):
                if verbose:
                    print(numeroGroupe, end = ' ')

                # Ensemble des documents présents dans ce groupe
                documentsDansK = np.arange(len(docs))[groupeDocuments.iloc[-1] == numeroGroupe]
                
                # Cas où il n'y a qu'un seul document dans le groupe
                if len(documentsDansK) == 1:
                    positionCentres.iloc[-1, numeroGroupe] = documentsDansK[0]
                    continue
                
                # Cas où l'on a toutes les distances 
                # (on ne s'embête pas avec une minimisation des calculs de WMD)
                elif np.all([get_distances(i, documentsDansK, True) != np.inf for i in documentsDansK]):

                    # On enregistre la somme des distances i -> j entre documents associé à un point de référence j
                    distanceTotale = np.array([np.sum(get_distances(j, documentsDansK)) for j in documentsDansK], dtype = typeData if typeData is not None else np.float32)

                # Cas où l'on n'a pas toutes les distances : on minimise le plus possible le calculs de distances WMD
                else:

                    # On enregistre la somme des distances i -> j entre documents associé à un point de référence j
                    # (initialisation)
                    distanceTotale = np.array([np.inf] * len(documentsDansK))
                    distanceNonTerminee = np.zeros(len(documentsDansK))
                    distancesRestantes = [documentsDansK] * len(documentsDansK)

                    # On va ensuite prendre chaque document comme référence. On regardera à chaque fois si le calcul
                    # de la distance ne dépasse pas la valeur minimale des distances totales calculées (auquel cas il ne 
                    # sert à rien de continuer à faire le calcul)
                    while np.min(distanceNonTerminee) < np.min(distanceTotale):
                        
                        # On récupère le centre le plus prometteur
                        # Pour cela on va récupérer la distance calculée jusqu'ici pour les différents documents références
                        # Si la distance est nulle (pas de calcul encore fait ou seule la distance doc-doc), on la laisse à 0
                        # Si la distance n'est pas nulle, on fait une moyenne : dActuelle * nbDocsGroupe / nbDistancesDejaAjouteees
                        # Afin d'avoir une estimation de la distance que l'on aurait à la fin.
                        temp = distanceNonTerminee.copy()
                        masque = [len(distancesRestantes[i]) < len(documentsDansK) for i in range(len(documentsDansK))]
                        temp[masque] = distanceNonTerminee[masque] * len(documentsDansK) / \
                            (len(documentsDansK) - np.array([len(distancesRestantes[i]) for i in range(len(documentsDansK))])[masque])
                        indiceMeilleurCentreActuel = np.argmin(temp)
                        meilleurCentreActuel = documentsDansK[indiceMeilleurCentreActuel]

                        # On ajoute à sa distance le premier point de sa liste
                        distanceNonTerminee[indiceMeilleurCentreActuel] += get_distance(meilleurCentreActuel, distancesRestantes[indiceMeilleurCentreActuel][0])
                        distancesRestantes[indiceMeilleurCentreActuel] = np.delete(distancesRestantes[indiceMeilleurCentreActuel], 0)

                        # Si sa liste est vide c'est que le point référence a sa distance complète
                        if len(distancesRestantes[indiceMeilleurCentreActuel]) == 0:
                            distanceTotale[indiceMeilleurCentreActuel] = distanceNonTerminee[indiceMeilleurCentreActuel]
                            distanceNonTerminee[indiceMeilleurCentreActuel] = np.inf

                        
                positionCentres.iloc[-1, numeroGroupe] = documentsDansK[np.argmin(distanceTotale)]

            if verbose:
                if len(positionCentres) >= 2:
                    print(") - Stabilité des centres : ", end = '')
                    for k in range(K):
                        print('V' if  positionCentres.iloc[-1, k] == positionCentres.iloc[-2, k] \
                            else 'X', end = ' ')

                print(f"({round(time() - t)} s)")

            # Arrêt express si les centres sont restés constants (pour éviter de recalculer les groupes)
            if len(groupeDocuments) >= 2 and np.all(positionCentres.iloc[-1] == positionCentres.iloc[-2]):
                break
    
    if np.any(positionCentres.iloc[-1] != positionCentres.iloc[-2]):
        print("Pas de convergence")

    nbDistancesCalculees = 0
    for dist in distancesDocs:
        nbDistancesCalculees += len(dist)

    # Calcul des variances intragroupe
    for k in range(K):
        intraVariances[k] = np.sum(np.array([get_distance(k, i) \
            for i in range(len(docs)) if groupeDocuments.iloc[-1][i] == k]) ** 2) / np.sum(groupeDocuments.iloc[-1] == k)

        
    return groupeDocuments, positionCentres, nbIter, nbDistancesCalculees / len(docs) ** 2, intraVariances
