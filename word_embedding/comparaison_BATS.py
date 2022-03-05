import numpy as np
import spacy
from deep_translator import GoogleTranslator
from gensim import models

nlp = spacy.load("fr_core_news_sm")
traducteur = GoogleTranslator(source='en',target='fr')

# Récupération des données BATS et passage au français
#vocabulaireCommun = set(vocabulaire).intersection(w2vecR.index_to_key)
def conv_BATS(pathBATS, listeFichiers, vocab: list[str]):
        BATS: list[list[str]] = []
        for nomFichier in listeFichiers:
            with open(pathBATS + nomFichier + ".txt") as f:
                fichier = f.readlines()
            for ligne in fichier:
                ligne = ligne.replace('/','\t').split('\t')
                ligne = ' '.join([traducteur.translate(s) for s in ligne])
                # passage en minuscules
                ligne = ligne.lower()
                #ligne = traducteur.translate(ligne)
                ligne = [token.lemma_ for token in nlp(ligne) if not token.is_stop and not token.is_punct]
                #ligne = [nlp(traducteur.translate(s)).lemma_ for s in ligne.split('\t')]
                for mot in ligne[1:]:
                    if ligne[0] in vocab and mot in vocab and mot != ligne[0] and (c := [ligne[0], mot]) not in BATS:
                        BATS.append(c)

        return BATS

def enregistrer_BATS(nomFichier: str, bats: str):
    lignes = [' '.join(s) + '\n' for s in bats]
    lignes[-1] = lignes[-1].replace('\n', '')
    with open(nomFichier, 'w') as f:
        f.writelines(lignes)

def set_BATS(vocabulaireCommun: list[str]):

    pathBATS2 = "../data/BATS_3.0/2_Derivational_morphology/"
    listeFichiers2 = ["D01 [noun+less_reg]","D02 [un+adj_reg]","D03 [adj+ly_reg]","D04 [over+adj_reg]","D05 [adj+ness_reg]",
    "D06 [re+verb_reg]","D07 [verb+able_reg]","D08 [verb+er_irreg]","D09 [verb+tion_irreg]","D10 [verb+ment_irreg]"]
    BATS2 = conv_BATS(pathBATS2, listeFichiers2, vocabulaireCommun)
    enregistrer_BATS("../data/BATS_3.0/2_fr.txt", BATS2)

    pathBATS3 = "../data/BATS_3.0/3_Encyclopedic_semantics/"
    listeFichiers3 = ["E01 [country - capital]", "E02 [country - language]", "E03 [UK_city - county]", "E04 [name - nationality]",
    "E05 [name - occupation]", "E06 [animal - young]", "E07 [animal - sound]", "E08 [animal - shelter]", "E09 [things - color]", "E10 [male - female]"]
    BATS3 = conv_BATS(pathBATS3, listeFichiers3, vocabulaireCommun)
    enregistrer_BATS("../data/BATS_3.0/3_fr.txt", BATS3)

def get_BATS() -> list[str]:
    pathBATS: str = "../data/BATS_3.0/"
    indices = ['2','3']
    BATS = []
    for s in indices:
        with open(pathBATS + s + "_fr.txt") as f:
            fichier = f.readlines()
            BATS += [l.replace('\n', '').split(' ') for l in fichier]
    return BATS



def get_stats_comparaisons_BATS(modele: models.KeyedVectors, reference: models.KeyedVectors):


    def get_position_proximite(modele: models.KeyedVectors, motCentral: str,
    mot2: str, listeMots: list[str] = None, distance: str = 'cosine') -> int:

        if listeMots is None:
            listeMots = modele.index_to_key

        listeMots = listeMots.copy()
        listeMots.remove(motCentral)
        distMots: list[float] = []
        pos = 1

        if distance == 'cosine':
            distMots = modele.distances(motCentral, listeMots)
            temp = sorted(distMots)
            pos = [temp.index(distMots[i]) for i in range(len(distMots)) if listeMots[i] == mot2][0] + 1
        
        return pos, pos / len(listeMots)

    # Pour chaque mot on détermine à quel point l'autre mot est loin
    # sa position dans le classement divisée par le nombre de mots du vocabulaire

    vocabulaire = list(set(modele.index_to_key).intersection(reference.index_to_key))

    BATS = get_BATS()
    n = len(BATS)
    stats = {}
    stats['taille_vocab'] = len(vocabulaire)
    stats['nb_comp'] = len(BATS) # Nombre de comparaisons
    delta = np.zeros(len(BATS))
    deltaDisCos = np.zeros(len(BATS))
    for i, couple in enumerate(BATS):
        if couple[0] in vocabulaire and couple[1] in vocabulaire:
            _, proxMod = get_position_proximite(   modele, couple[0], couple[1], vocabulaire, 'cosine')
            _, proxRef = get_position_proximite(reference, couple[0], couple[1], vocabulaire, 'cosine')
            delta[i] = proxMod - proxRef
            cosMod = modele.distance(couple[0], couple[1])
            cosRef = reference.distance(couple[0], couple[1])
            deltaDisCos[i] = cosMod - cosRef
        else:
            n -= 1
    
    # RMSE de la fréquence
    stats['rmse_freq'] = np.linalg.norm(delta) / np.sqrt(n)

    # Erreur moyenne
    stats['err_moy_freq'] = np.sum(delta) / n

    # Erreur distance cos
    stats['err_dis_cos'] = np.sum(deltaDisCos) / n

    # RMSE distance dos
    stats['rmse_dis_cos'] =  np.linalg.norm(deltaDisCos) / np.sqrt(n)

    return stats
