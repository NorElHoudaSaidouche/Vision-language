"""
TP 5 - Vision et language
Implémentation des fonctions
Fait par : SAIDOUCHE Nor El Houda & HANACHI Ourida
"""
# Importation des bibliothèques
from requests import get
from keras.models import model_from_yaml
import numpy as np


def download(url, file_name):
    # open in binary mode
    with open(file_name, "wb") as file:
        # get request
        response = get(url)
        # write to file
        file.write(response.content)


def save_model(model, savename):
    # serialize model to YAML
    model_yaml = model.to_yaml()
    with open(savename + ".yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)
    print("Yaml Model ", savename, ".yaml saved to disk")
    # serialize weights to HDF5
    model.save_weights(savename + ".h5")
    print("Weights ", savename, ".h5 saved to disk")


def load_model(savename):
    with open(savename + ".yaml", "r") as yaml_file:
        model = model_from_yaml(yaml_file.read())
    print("Yaml Model ", savename, ".yaml loaded ")
    model.load_weights(savename + ".h5")
    print("Weights ", savename, ".h5 loaded ")
    return model


def sampling(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    predsN = pow(preds, 1.0 / temperature)
    predsN /= np.sum(predsN)
    probas = np.random.multinomial(1, predsN, 1)
    return np.argmax(probas)
