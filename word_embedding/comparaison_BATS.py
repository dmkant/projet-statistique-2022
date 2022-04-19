import numpy as np
import spacy
from deep_translator import GoogleTranslator
from gensim import models
import os
import pandas as pd

nlp = spacy.load("fr_core_news_md")
traducteur = GoogleTranslator(source='en',target='fr')
pathBATS: str = "data/BATS_3.0/"

# Récupération des données BATS et passage au français
#vocabulaireCommun = set(vocabulaire).intersection(w2vecR.index_to_key)
def conv_BATS(pathBATS, listeFichiers, vocab: list[str]):
        BATS: list[list[str]] = []
        for nomFichier in listeFichiers:
            print(nomFichier)
            with open(pathBATS + nomFichier + ".txt") as f:
                fichier = f.readlines()
            for ligne in fichier:

                ligne = ligne.replace('\n','').replace('/','\t')
                ligne = ligne.split('\t')
                res = []
                for mot in ligne:
                    try:
                        trad = traducteur.translate(mot.replace('_', ' '))
                        # Si la traduction est en un seul mot
                        if ' ' not in trad:
                            res.append(trad)
                    except:
                        pass

                if len(ligne) > 1:
                    ligne = ' '.join(res)
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

    pathBATS2 = pathBATS + "2_Derivational_morphology/"
    listeFichiers2 = ["D01 [noun+less_reg]","D02 [un+adj_reg]","D03 [adj+ly_reg]","D04 [over+adj_reg]","D05 [adj+ness_reg]",
    "D06 [re+verb_reg]","D07 [verb+able_reg]","D08 [verb+er_irreg]","D09 [verb+tion_irreg]","D10 [verb+ment_irreg]"]
    BATS2 = conv_BATS(pathBATS2, listeFichiers2, vocabulaireCommun)
    enregistrer_BATS(pathBATS + "2_fr.txt", BATS2)

    pathBATS3 = pathBATS + "3_Encyclopedic_semantics/"
    listeFichiers3 = ["E01 [country - capital]", "E02 [country - language]", "E03 [UK_city - county]", "E04 [name - nationality]",
    "E05 [name - occupation]", "E06 [animal - young]", "E07 [animal - sound]", "E08 [animal - shelter]", "E09 [things - color]", "E10 [male - female]"]
    BATS3 = conv_BATS(pathBATS3, listeFichiers3, vocabulaireCommun)
    enregistrer_BATS(pathBATS + "3_fr.txt", BATS3)

    pathBATS4 = pathBATS + "4_Lexicographic_semantics/"
    listeFichiers4 = ["L01 [hypernyms - animals]", "L02 [hypernyms - misc]", "L03 [hyponyms - misc]", "L04 [meronyms - substance]",
    "L05 [meronyms - member]", "L06 [meronyms - part]", "L07 [synonyms - intensity]", "L08 [synonyms - exact]",
    "L09 [antonyms - gradable]", "L10 [antonyms - binary]"]
    BATS4 = conv_BATS(pathBATS4, listeFichiers4, vocabulaireCommun)
    enregistrer_BATS(pathBATS + "4_fr.txt", BATS4)

def get_BATS() -> list[str]:
    pathBATS: str = "data/BATS_3.0/"
    indices = ['2','3','4']
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


def evaluation_BATS():
    #Tuning parameters
    list_models_filename = os.listdir("data/training_models")
    list_windows = []
    list_dim_emb = []
    list_type_model = []
    #Evaluation metrics
    list_ref_err_dis_cos = []
    list_ref_rmse_dis_cos = []
    list_ref_err_moy_freq = []
    list_ref_rmse_freq = []

    # Bats evaluation
    modeleReference: models.KeyedVectors = models.KeyedVectors.load_word2vec_format("data/frWiki_no_phrase_no_postag_1000_skip_cut100.bin", 
                                                                        binary=True, unicode_errors="ignore")

    for models_filename in list_models_filename:
        embed_model = models.KeyedVectors.load_word2vec_format(f"data/training_models/{models_filename}")
        tune_param = models_filename.split("_")
        list_type_model.append(tune_param[0])
        list_windows.append(tune_param[1])
        list_dim_emb.append(tune_param[2].split(".")[0])


        #Reference evaluation
        print(models_filename,": ",list_models_filename.index(models_filename)," : BATS",end="\r")
        stats = get_stats_comparaisons_BATS(embed_model, modeleReference)
        list_ref_err_dis_cos.append(stats["err_dis_cos"])
        list_ref_rmse_dis_cos.append(stats["rmse_dis_cos"])
        list_ref_err_moy_freq.append(stats["err_moy_freq"])
        list_ref_rmse_freq.append(stats["rmse_freq"])

    df_evaluation = pd.DataFrame(list(zip(
        list_models_filename, list_type_model, list_windows, list_dim_emb,
        list_ref_err_dis_cos,list_ref_rmse_dis_cos,list_ref_err_moy_freq,list_ref_rmse_freq)),
                                    columns=[ "models_filename", "type_model", "windows", "dim_emb",
        "ref_err_dis_cos","ref_rmse_dis_cos","ref_err_moy_freq","ref_rmse_freq"])
    df_evaluation.to_csv("data/tuning/evaluation_bats.csv",sep=";",index=False)


if __name__ == '__main__':
    with open("data/liste_lemmes.txt") as f:
        v = f.readlines()
        v = [s.replace('\n', '') for s in v]

    set_BATS(v)
    
