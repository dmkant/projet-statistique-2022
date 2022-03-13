import numpy as np
from pandas import DataFrame, Series


def stats_comparaison_clusters(docs: list[list[str]], clusters: list[int], verbose: bool = True):

    if len(docs) != len(clusters):
        raise ValueError("La taille du corpus et le nombre de labels ne correspond pas")

    nomsClusters = sorted(list(set(clusters)))
    nomsColonnes: list = ['cluster', 'mot', 'occur_cluster', 'freq_cluster', 'variations']
    donneesClusters: DataFrame = DataFrame(columns = nomsColonnes)
    # Recherche des mots présents dans chaque cluster, calcul des occurences 
    # et de la fréquence de chacuns
    for k in nomsClusters:
        docsCluster: list[list[str]] = [docs[i] for i in range(len(clusters)) if clusters[i] == k]
        motsCluster = {}
        for doc in docsCluster:
            for mot in doc:
                if mot in motsCluster:
                    motsCluster[mot] += 1
                else:
                    motsCluster[mot] = 1
        

        nbMots = sum(motsCluster.values())
        for mot, nbOccurences in motsCluster.items():
            ligne: Series = Series([k, mot, nbOccurences, 100 * nbOccurences / nbMots, np.inf], index = nomsColonnes)
            donneesClusters = donneesClusters.append(ligne, ignore_index = True)
    
    if verbose:
        # Affichage des plus grandes fréquences dans chaque cluster
        N: int = 5 # Nombre de mots à afficher par cluster
        #print(f"N = {N}")
        print("Mots les plus fréquents")
        print("Clefs : B - Fréquence la + importante parmi tous les clusters | X : Terme présent plusieurs fois dans le top")
        nomsColonnes2: list = ['mot', 'cluster', 'freq_cluster', 'plus_forte_freq', 'utilisePlusieursFois']
        donneesRepresentation: DataFrame = DataFrame(columns = nomsColonnes2)

        for cluster in nomsClusters:
            
            listeFrequences = donneesClusters.loc[donneesClusters['cluster'] == cluster, ['mot','freq_cluster']].copy()
            listeFrequences.sort_values(by = 'freq_cluster', ascending = False, inplace = True, ignore_index = True)
            

            for i in range(N):
                mot: str =  listeFrequences.loc[i, 'mot']
                freq: float = listeFrequences.loc[i, 'freq_cluster']
                maxFreqTousClusters: float = donneesClusters[donneesClusters["mot"] == mot].freq_cluster.max()
                ligne: Series = Series([mot, cluster, freq, freq == maxFreqTousClusters, False],
                 index = nomsColonnes2)
                donneesRepresentation = donneesRepresentation.append(ligne, ignore_index = True)
                
        
        # On regarde si le mot est présent plusieurs fois dans le top
        for mot in donneesRepresentation.mot.unique():
            if donneesRepresentation.mot.value_counts()[mot] > 1:
                donneesRepresentation.loc[donneesRepresentation.mot == mot, 'utilisePlusieursFois'] = True

        for cluster in nomsClusters:
            
            print(f"cluster {cluster} - ", end = '')

            data: DataFrame = donneesRepresentation[donneesRepresentation.cluster == cluster].copy()
            data.sort_values(by = 'freq_cluster', ascending = False, inplace = True, ignore_index = True)

            for i in range(N):
                ligne = data.loc[i]
                freq: float = ligne.freq_cluster
                mot: str = ligne.mot
                #print(freq, maxFreqTousClusters)
                clefs: str = ''
                clefs += 'B' if freq == maxFreqTousClusters else ''
                clefs += 'X' if ligne.utilisePlusieursFois else ''
                print(f"{mot} ({freq} %) [{clefs}]",
                end = ', ' if i != N - 1 else '\n')


    # Calcul de la 'variance' de chaque mot afin de trouver quel 
    # mot discrimine le plus sur chaque cluster
    for mot in donneesClusters['mot'].unique():

        masque = donneesClusters['mot'] == mot
        clustersConcernes: Series = donneesClusters.loc[masque, 'cluster']

        if len(clustersConcernes) == 1:
            donneesClusters.loc[masque, 'variations'] = 0.

        else:
            for cluster in clustersConcernes:
                ##print(np.any((donneesClusters.cluster == cluster) & (donneesClusters.mot == mot)))
                valeurRef: float = donneesClusters.loc[np.logical_and(donneesClusters.cluster == cluster, donneesClusters.mot == mot), 'freq_cluster'].to_numpy()[0]
                ##print(valeurRef, donneesClusters.loc[donneesClusters.mot == mot, 'freq_cluster'])
                var: float = np.sum((donneesClusters.loc[donneesClusters.mot == mot, 'freq_cluster'].to_numpy() - valeurRef)**2) / len(clustersConcernes)
                ##print(var)
                donneesClusters.loc[(donneesClusters.cluster == cluster) & (donneesClusters.mot == mot), 'variations'] = var

    if verbose:
        # Affichage des mots où l'on retrouve la plus forte variation inter-cluster
        # Le terme doit avoir la plus forte variation et être plus présent dans ce cluster
        # que dans les autres 
        print("Mots les plus discriminants")

        print("En général : ", end = '')
        listeVar = donneesClusters[['mot', 'variations']].copy()
        listeVar.sort_values(by = 'variations', ascending = False, inplace = True, ignore_index = True)

        motsUtilises = []
        i: int = 0
        while len(motsUtilises) < N and i < len(listeVar):
            ligne = listeVar.iloc[i]
            #print(i, ligne.mot)
            if ligne.mot not in motsUtilises:
                print(f"{ligne.mot} ({ligne.variations})", end = ', ' if len(motsUtilises) != N - 1 else '\n')
                motsUtilises.append(ligne.mot)
            i += 1

        del listeVar
        for cluster in nomsClusters:
            
            print(f"cluster {cluster} - ", end = '')
            listeFrequences = donneesClusters.loc[donneesClusters['cluster'] == cluster, ['mot', 'freq_cluster', 'variations']].copy()
            listeFrequences.sort_values(by = 'variations', ascending = False, inplace = True, ignore_index = True)
            

            for i in range(N):
                print(f"{listeFrequences.loc[i, 'mot']} ({listeFrequences.loc[i, 'variations']})", end = ', ' if i != N - 1 else '\n')

            
            i: int = 0
            nbMots = 0
            while nbMots < N and i < len(listeFrequences):

                mot: str = listeFrequences.loc[i, 'mot']
                if listeFrequences.loc[i, 'freq_cluster'] == donneesClusters.loc[donneesClusters.mot == 'mot'].freq_cluster.max():
                    
                    print(f"{listeFrequences.loc[i, 'mot']} ({listeFrequences.loc[i, 'variations']})", end = ', ' if nbMots != N - 1 else '\n')
                    nbMots += 1
                
                i += 1
            

    return donneesClusters
