"""
TP 5 - Vision et language
Exercice 1 : Simplification du vocabulaire
Fait par : SAIDOUCHE Nor El Houda & HANACHI Ourida
"""
# Importation des bibliothèques
from divers import *
import pandas as pd
import _pickle as pickle
import matplotlib.pyplot as plt
import numpy as np

# Téléchargement du dataset
download('http://cedric.cnam.fr/~thomen/cours/US330X/flickr_8k_train_dataset.txt', "flickr_8k_train_dataset.txt")

# Chargement des données d'entraînement
filename = 'flickr_8k_train_dataset.txt'
df = pd.read_csv(filename, delimiter='\t')
nb_samples = df.shape[0]
iter_w = df.iterrows()

bow = {}
nbwords = 0

for i in range(nb_samples):
    x = iter_w.__next__()
    # Diviser la légende en mots
    cap_words = x[1][1].split()
    # Supprimer les majuscules
    cap_wordsl = [w.lower() for w in cap_words]
    nbwords += len(cap_wordsl)
    for w in cap_wordsl:
        if w in bow:
            bow[w] = bow[w] + 1
        else:
            bow[w] = 1

bown = sorted([(value, key) for (key, value) in bow.items()], reverse=True)

# Calcul de la fréquence cumulée des 1000 premiers mots conservés
nbkeep = 1000
freqnc = np.cumsum([float(w[0]) / nbwords * 100.0 for w in bown])
print("number of kept words=" + str(nbkeep) + " - ratio=" + str(freqnc[nbkeep - 1]) + " %")

# Affichage de la liste des 100 premiers mots
x_axis = [str(bown[i][1]) for i in range(100)]
plt.figure(dpi=300)
plt.xticks(rotation=90, fontsize=3)
plt.ylabel('Word Frequency')
plt.bar(x_axis, freqnc[0:100])
plt.show()

# Téléchargement du dataset
download('http://cedric.cnam.fr/~thomen/cours/US330X/Caption_Embeddings.p', "Caption_Embeddings.p")

outfile = 'Caption_Embeddings.p'
[listwords, embeddings] = pickle.load(open(outfile, "rb"))

embeddings_new = np.zeros((nbkeep, 102))
listwords_new = []

for i in range(nbkeep):
    listwords_new.append(bown[i][1])
    embeddings_new[i, :] = embeddings[listwords.index(bown[i][1]), :]
    embeddings_new[i, :] /= np.linalg.norm(embeddings_new[i, :])

listwords = listwords_new
embeddings = embeddings_new
outfile = "Caption_Embeddings_" + str(nbkeep) + ".p"

with open(outfile, "wb") as pickle_f:
    pickle.dump([listwords, embeddings], pickle_f)
