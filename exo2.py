"""
TP 5 - Vision et language
Exercice 2 : Création des données d’apprentissage et de test
Fait par : SAIDOUCHE Nor El Houda & HANACHI Ourida
"""
# Importation des bibliothèques
import _pickle as pickle
import pandas as pd
from divers import *


# Données d'entraînement
nbkeep = 1000
outfile = "Caption_Embeddings_" + str(nbkeep) + ".p"
[listwords, embeddings] = pickle.load(open(outfile, "rb"))
indexwords = {}
for i in range(len(listwords)):
    indexwords[listwords[i]] = i

filename = 'flickr_8k_train_dataset.txt'
df = pd.read_csv(filename, delimiter='\t')
nbTrain = df.shape[0]
iter_w = df.iterrows()
# Ensemble de légendes
caps = []
# Ensemble d'images
imgs = []
for i in range(nbTrain):
    x = iter_w.__next__()
    caps.append(x[1][1])
    imgs.append(x[1][0])

maxLCap = 0

for caption in caps:
    l = 0
    words_in_caption = caption.split()
    for j in range(len(words_in_caption) - 1):
        current_w = words_in_caption[j].lower()
        if current_w in listwords:
            l += 1
        if l > maxLCap:
            maxLCap = l

print("max caption length =" + str(maxLCap))

# Chargement des features des images
download('http://cedric.cnam.fr/~thomen/cours/US330X/encoded_images_PCA.p', "encoded_images_PCA.p")
encoded_images = pickle.load(open("encoded_images_PCA.p", "rb"))

indexwords = {}
for i in range(len(listwords)):
    indexwords[listwords[i]] = i

tinput = 202

# Ignorer START : nous ne pourrons jamais prédire <start> .
tVocabulary = len(listwords)

X_train = np.zeros((nbTrain, maxLCap, tinput))
Y_train = np.zeros((nbTrain, maxLCap, tVocabulary), bool)

ll = 50
nbtot = 0
nbkept = 0

for i in range(nbTrain):
    words_in_caption = caps[i].split()

    nbtot += len(words_in_caption) - 1
    indseq = 0
    for j in range(len(words_in_caption) - 1):

        current_w = words_in_caption[j].lower()

        if j == 0 and current_w != '<start>':
            print("PROBLEM")
        if current_w in listwords:
            X_train[i, indseq, 0:100] = encoded_images[imgs[i]]
            X_train[i, indseq, 100:202] = embeddings[listwords.index(current_w), :]

        next_w = words_in_caption[j + 1].lower()

        index_pred = 0
        if next_w in listwords:
            nbkept += 1
            index_pred = indexwords[next_w]
            Y_train[i, indseq, index_pred] = True

            indseq += 1

outfile = 'Training_data_' + str(nbkeep)
# Sauvegarde du Tensor
np.savez(outfile, X_train=X_train, Y_train=Y_train)

# Données Test
download('http://cedric.cnam.fr/~thomen/cours/US330X/flickr_8k_test_dataset.txt', "flickr_8k_test_dataset.txt")
filename = 'flickr_8k_test_dataset.txt'
df = pd.read_csv(filename, delimiter='\t')
nbTest = df.shape[0]
iter_w = df.iterrows()
# Ensemble de légendes
caps = []
# Ensemble d'images
imgs = []
for i in range(nbTest):
    x = iter_w.__next__()
    caps.append(x[1][1])
    imgs.append(x[1][0])

indexwords = {}
for i in range(len(listwords)):
    indexwords[listwords[i]] = i

# Chargement des features des images
encoded_images = pickle.load(open("encoded_images_PCA.p", "rb"))

tVocabulary = len(listwords)
X_test = np.zeros((nbTest, maxLCap, tinput))
Y_test = np.zeros((nbTest, maxLCap, tVocabulary), bool)

for i in range(nbTest):
    words_in_caption = caps[i].split()
    indseq = 0
    for j in range(len(words_in_caption) - 1):
        current_w = words_in_caption[j].lower()
        if current_w in listwords:
            X_test[i, indseq, 0:100] = encoded_images[imgs[i]]
            X_test[i, indseq, 100:202] = embeddings[listwords.index(current_w), :]

        next_w = words_in_caption[j + 1].lower()
        if next_w in listwords:
            index_pred = indexwords[next_w]
            Y_test[i, indseq, index_pred] = True
            indseq += 1

outfile = 'Test_data_' + str(nbkeep)
# Sauvegarde du tensor
np.savez(outfile, X_test=X_test, Y_test=Y_test)
