"""
TP 5 - Vision et language
Exercice 4 : Évaluation du modèle
Fait par : SAIDOUCHE Nor El Houda & HANACHI Ourida
"""
# Importation des bibliothèques
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import _pickle as pickle
import pandas as pd
from divers import *
from keras.optimizers import Adam

# Chargement du modèle
nameModel = 'modele_exo5'
model = load_model(nameModel)

optim = Adam()
model.compile(loss="categorical_crossentropy", optimizer=optim, metrics=['accuracy'])

# Chargement des données test
nbkeep = 1000
outfile = 'Test_data_' + str(nbkeep) + '.npz'
Test_data = np.load(outfile)

X_test = Test_data['X_test']
Y_test = Test_data['Y_test']

outfile = "Caption_Embeddings_" + str(nbkeep) + ".p"
[listwords, embeddings] = pickle.load(open(outfile, "rb"))
indexwords = {}
for i in range(len(listwords)):
    indexwords[listwords[i]] = i

# Affichage d'une image alétaoire de l'ensemble test
ind = np.random.randint(X_test.shape[0])
filename = 'flickr_8k_test_dataset.txt'
df = pd.read_csv(filename, delimiter='\t')
iter_w = df.iterrows()

for i in range(ind + 1):
    x = iter_w.__next__()

imname = x[1][0]
print("image name=" + imname + " caption=" + x[1][1])
dirIm = "./Flicker8k_Dataset/"

img = mpimg.imread(dirIm + imname)
plt.figure(dpi=100)
plt.imshow(img)
plt.axis('off')
plt.show()

# Prédiction
pred = model.predict(X_test[ind:ind + 1, :, :])

# Affichage des prédictions
nbGen = 5
temperature = 0.1
for s in range(nbGen):
    wordpreds = "Caption n° " + str(s + 1) + ": "
    indpred = sampling(pred[0, 0, :], temperature)
    wordpred = listwords[indpred]
    wordpreds += str(wordpred) + " "
    X_test[ind:ind + 1, 1, 100:202] = embeddings[indpred]
    cpt = 1
    while str(wordpred) != '<end>' and cpt < 30:
        pred = model.predict(X_test[ind:ind + 1, :, :])
        indpred = sampling(pred[0, cpt, :], temperature)
        wordpred = listwords[indpred]
        wordpreds += str(wordpred) + " "
        cpt += 1
        X_test[ind:ind + 1, cpt, 100:202] = embeddings[indpred]

    print(wordpreds)
