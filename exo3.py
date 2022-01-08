"""
TP 5 - Vision et language
Exercice 3 : Entraînement du modèle
Fait par : SAIDOUCHE Nor El Houda & HANACHI Ourida
"""
# Importation des bibliothèques
from keras.layers.recurrent import SimpleRNN
from keras.models import Sequential
from keras.layers import Dense, Activation, Masking
from keras.optimizers import Adam

from divers import *

SEQLEN = 35
taille_chars = 202
HSIZE = 100
model = Sequential()
model.add(Masking(mask_value=0.0, input_shape=(SEQLEN, taille_chars)))
model.add(SimpleRNN(HSIZE, return_sequences=True, input_shape=(SEQLEN, taille_chars), unroll=True))
model.add(Dense(1000, name='fc1'))
model.add(Activation("softmax"))
model.summary()
nbkeep = 1000

Train_data = np.load('Training_data_' + str(nbkeep) + '.npz')
X_train = Train_data['X_train']
Y_train = Train_data['Y_train']

Test_data = np.load('Test_data_' + str(nbkeep) + '.npz')
X_test = Test_data['X_test']
Y_test = Test_data['Y_test']

BATCH_SIZE = 10
NUM_EPOCHS = 10
optim = Adam()
model.compile(loss="categorical_crossentropy", optimizer=optim, metrics=['accuracy'])
model.summary()

model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS)

scores_train = model.evaluate(X_train, Y_train, verbose=1)
scores_test = model.evaluate(X_test, Y_test, verbose=1)
print("PERFS TRAIN: %s: %.2f%%" % (model.metrics_names[1], scores_train[1] * 100))
print("PERFS TEST: %s: %.2f%%" % (model.metrics_names[1], scores_test[1] * 100))

# Sauvegadre du modele
nameModel = 'modele_exo5'
save_model(model, nameModel)

