# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 19:38:53 2019

@author: Julnor Georges, Rose-Camille Vincent
"""



##############################################################################
#
#    ALGORITHMIE DU BIG DATA
#
##############################################################################


#
# QUESTION 0 - IMPORTATION DES PACKAGES ET LIBRAIRIES UTILISEES PAR LA SUITE
# 

import pandas as pd
import numpy as np
import csv as csv
import dask
import dask.dataframe as dd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


#
# QUESTION 1 - IMPORT DU JEU DE DONNEES
# 


### Q1.1 - Indiquer le dossier et le fichier cible

dossier = "/Users/julnorgeorges/Documents/Toulouse/Big data/TP2/"
train_ech = "train_echantillon.csv"
train = "train.csv"
chemin_ech = dossier + train_ech
chemin = dossier + train


### Q1.2 - Importer les jeux de données complets et échantillonnés
###        Prediction du prix du taxi à New York - https://www.kaggle.com/c/new-york-city-taxi-fare-prediction/data


# ---------- Utiliser une librairie usuelle (version de fichier échantillonnée)

set_ech = pd.read_csv(chemin_ech)


# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory) (version complète du fichier)
train_set = dd.read_csv(chemin)



#
# QUESTION 2 - PREPARATION/NETTOYAGE DU JEU DE DONNEES
# 


### Q2.1 - Nettoyer et préparer les données


# Enlever les valeurs incorrectes ou manquantes (si pertinent)


# ---------- Utiliser une librairie usuelle
set_ech.info()
set_ech.isnull().any()
set_ech.isnull().sum().sum()
# On constate qu'il y a des valeurs manquantes pour certaines variables de 
# géolocalisation (dorpoff_longitude, dropoff_latitude). On peut enlever les
# valeurs manquantes.

set_ech_clean = set_ech.dropna()

# On pense que le coût de le course ne peut être négatif, on va remplacer les
# valeurs négatives par zéro

set_ech_clean[set_ech_clean['fare_amount']<0]=0

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)
train_set_clean = train_set.dropna()
train_set_clean.loc[train_set_clean['fare_amount']>0] 


# Ne garder que les variables de géolocalisation (pour le jeu de données en entrée) et
# la variable "fare_amount" pour la sortie
variable_keep = ["fare_amount","pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"]

# ---------- Utiliser une librairie usuelle
input_var = variable_keep.copy()
input_var.remove("fare_amount")
X,y = set_ech_clean[input_var], set_ech_clean["fare_amount"]

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

X_d, y_d = train_set_clean[input_var], train_set_clean["fare_amount"]


# Obtenir les caractéristiques statistiques de base des variables d'entrée et de sortie
# (par exemple, min, moyenne, mdéiane, max) et filter les valeurs aberrantes


# ---------- Utiliser une librairie usuelle
print("***** Statistiques de base : sample****")
print(X.pickup_longitude.describe())
print(X.pickup_latitude.describe())
print(X.dropoff_longitude.describe())
print(X.dropoff_latitude.describe())
print(y.describe())


# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)
# pickup_longitude statistiques de base
moyenne, minimum, maximum = dask.compute(X_d.pickup_longitude.mean(), X_d.pickup_longitude.min(), X_d.pickup_longitude.max())

print("***** Statistiques de base : Big data*****")
médiane = X_d.pickup_longitude.compute().quantile([.5])
print(moyenne, minimum, maximum, médiane)

# pickup_latitude statistiques de base
moyenne, minimum, maximum = dask.compute(X_d.pickup_latitude.mean(), X_d.pickup_latitude.min(), X_d.pickup_latitude.max())

médiane = X_d.pickup_latitude.compute().quantile([.5])

print(moyenne, minimum, maximum, médiane)

# dropoff_longitude statistiques de base
moyenne, minimum, maximum = dask.compute(X_d.dropoff_longitude.mean(), X_d.dropoff_longitude.min(), X_d.dropoff_longitude.max())

médiane = X_d.dropoff_longitude.compute().quantile([.5])

print(moyenne, minimum, maximum, médiane)

# dropoff_latitude statistiques de base
moyenne, minimum, maximum = dask.compute(X_d.dropoff_latitude.mean(), X_d.dropoff_latitude.min(), X_d.dropoff_latitude.max())

médiane = X_d.dropoff_latitude.compute().quantile([.5])

print(moyenne, minimum, maximum, médiane)

# Fare_amount statistiques de base
print("***** Statistiques de base: Fare_amount*****")
moyenne, minimum, maximum = y_d.compute(y.mean(), y.min(), y.max())
médiane = y_d.compute().quantile([.5])
print(moyenne, minimum, maximum, médiane)

# Visualiser les distributions des variables d'entrée et de sortie (histogramme, pairplot)

# ---------- Utiliser une librairie usuelle

for var in input_var:
    X[var].plot.hist()
    plt.title(var)
    plt.show()
    
y.plot.hist()
plt.title("fare_amount")

sns.pairplot(X)

# Séparer la variable à prédire ("fare_amount") des autres variables d'entrée
# Créer un objet avec variables d'entrée et un objet avec valeurs de sortie (i.e. "fare_amount")



# ---------- Utiliser une librairie usuelle

X, y = set_ech_clean[input_var], set_ech_clean["fare_amount"] 

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

X_d, y_d = train_set_clean[input_var], train_set_clean["fare_amount"]


# Standardiser la matrice d'entrée et les vecteurs de sortie (créer un nouvel objet)


# ---------- Utiliser une librairie usuelle
X_scaled = StandardScaler().fit_transform(X)
y_scaled = preprocessing.scale(y)


# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

X_d_scaled = StandardScaler().fit_transform(X_d)
y_d_scaled = preprocessing.scale(y_d)




#
# QUESTION 3 - CLUSTERING DU JEU DE DONNEES
# 




### Q3.1 - Réaliser un clustering k-means sur les données d'entrée standardisées


# ---------- Utiliser une librairie usuelle
kmeans_m1 = KMeans(n_clusters = 4, init = 'k-means++', max_iter=1000, n_init = 100, random_state=0).fit(X_scaled)

kmeans_model = KMeans(n_clusters = 5, random_state = 0).fit(X_scaled)
labels = kmeans_model.labels_
inertia = kmeans_model.inertia_
print("nombre de clusters : "  +str(labels)+ "  -inertie : " +str(inertia))


labels = kmeans_m1.labels_
inertia = kmeans_m1.inertia_
print("nombre de clusters : "  +str(labels)+ "  -inertie : " +str(inertia))

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

from dask.cluster import KMeans
kmeans_dask = KMeans(n_clusters = 4)
kmeans_dask.fit_transform(X_d_scaled)
cluster=kmeans_dask.labels



### Q3.2 - Tracer la figure de l'inertie intraclusters et du R² en fonction du nombre de  clusters


# ---------- Utiliser une librairie usuelle

#CODE





### Q3.3 - A partir de combien de clusters on peut dire que partitionner n'apporte plus 
###        grand chose? Pourquoi?



#REPONSE ECRITE (3 lignes maximum)





### Q3.4 - Comment pouvez-vous qualifier les clusters obtenus selon les variables originales?
###        Par exemple, y a-t-il des clusters selon la localisation ? 




#REPONSE ECRITE (3 lignes maximum)



### Q3.5 - Visualiser les clusters avec des couleurs différentes sur un 'pairplot' avec plusieurs variables


# ---------- Utiliser une librairie usuelle


#CODE









#
# QUESTION 4 - ANALYSE EN COMPOSANTES PRINCIPALES (ACP) POUR SIMPLIFIER LE JEU DE DONNEES
# 



### Q4.1 - Faire une ACP sur le jeu de données standardisé


# ---------- Utiliser une librairie usuelle

#CODE

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

#CODE


### Q4.2 - Réaliser le diagnostic de variance avec un graphique à barre (barchart)

 

# ---------- Utiliser une librairie usuelle


#CODE




### Q4.3 - Combien de composantes doit-on garder? Pourquoi?
       


#REPONSE ECRITE (3 lignes maximum)




### Q4.4 - Tracer un graphique 'biplot' indiquant les variables initiales selon les 2 premières CP
###        Sélectionner éventuellement un sous-échantillon de points pour faciliter la visualisation

 

# ---------- Utiliser une librairie usuelle


#CODE




### Q4.5 - Comment les variables initiales se situent-elles par rapport aux 2 premières CP? 


#REPONSE ECRITE (3 lignes maximum)











#
# QUESTION 5 - REGRESSION LINEAIRE
# 



### Q5.1 - Mener une régression linéaire de la sortie "fare_amount" 
###        en fonction de l'entrée (mise à l'échelle), sur tout le jeu de données


# ---------- Utiliser une librairie usuelle

#CODE

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

#CODE


### Q5.2 - Que pouvez-vous dire des résultats du modèle? Quelles variables sont significatives?



#REPONSE ECRITE (3 lignes maximum)



### Q5.3 - Prédire le prix de la course en fonction de nouvelles entrées avec une régression linéaire


# Diviser le jeu de données initial en échantillons d'apprentissage (60% des données), validation (20%) et test (20%)


# ---------- Utiliser une librairie usuelle

#CODE

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

#CODE


# Réaliser la régression linéaire sur l'échantillon d'apprentissage, tester plusieurs valeurs
# de régularisation (hyperparamètre de la régression linéaire) et la qualité de prédiction sur l'échantillon de validation. 


# ---------- Utiliser une librairie usuelle

#CODE

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

#CODE

# Calculer le RMSE et le R² sur le jeu de test.



# ---------- Utiliser une librairie usuelle

#CODE

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

#CODE

# Quelle est la qualité de la prédiction sur le jeu de test ?


#REPONSE ECRITE (3 lignes maximum)








#
# QUESTION 6 - REGRESSION LOGISTIQUE
# 



### Q6.1 - Mener une régression logisitique de la sortie "fare_amount" (après binarisation selon la médiane) 
###        en fonction de l'entrée (mise à l'échelle), sur tout le jeu de données


# Créer la sortie binaire 'fare_binaire' en prenant la valeur médiane de "fare_amount" comme seuil


# ---------- Utiliser une librairie usuelle

#CODE

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

#CODE


# Mener la régression logistique de "fare_binaire" en fonction des entrées standardisées


# ---------- Utiliser une librairie usuelle

#CODE

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

#CODE




### Q6.2 - Que pouvez-vous dire des résultats du modèle? Quelles variables sont significatives?



#REPONSE ECRITE (3 lignes maximum)



### Q6.3 - Prédire la probabilité que la course soit plus élevée que la médiane
#           en fonction de nouvelles entrées avec une régression linéaire


# Diviser le jeu de données initial en échantillons d'apprentissage (60% des données), validation (20%) et test (20%)


# ---------- Utiliser une librairie usuelle

#CODE

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

#CODE


# Réaliser la régression logistique sur l'échantillon d'apprentissage et en testant plusieurs valeurs
# de régularisation (hyperparamètre de la régression logistique) sur l'échantillon de validation. 


# ---------- Utiliser une librairie usuelle

#CODE

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

#CODE


# Calculer la précision (accuracy) et l'AUC de la prédiction sur le jeu de test.



# ---------- Utiliser une librairie usuelle

#CODE

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

#CODE


# Quelle est la qualité de la prédiction sur le jeu de test ?


#REPONSE ECRITE (3 lignes maximum)







#
# QUESTION 7 - RESEAU DE NEURONES (QUESTION BONUS)
# 



### Q7.1 - Mener une régression de la sortie "fare_amount" en fonction de l'entrée (mise à l'échelle), 
###       sur tout le jeu de données, avec un réseau à 2 couches cachées de 10 neurones chacune



# ---------- Utiliser une librairie usuelle

#CODE

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

#CODE



### Q7.2 - Prédire le prix de la course en fonction de nouvelles entrées avec le réseau de neurones entraîné


# Diviser le jeu de données initial en échantillons d'apprentissage (60% des données), validation (20%) et test (20%)


# ---------- Utiliser une librairie usuelle

#CODE

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

#CODE


# Réaliser la régression avec réseau de neurones sur l'échantillon d'apprentissage et en testant plusieurs 
# nombre de couches et de neurones par couche sur l'échantillon de validation. 


# ---------- Utiliser une librairie usuelle

#CODE

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

#CODE


# Calculer le RMSE et le R² de la meilleure prédiction sur le jeu de test.


# ---------- Utiliser une librairie usuelle

#CODE

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

#CODE

# Quelle est la qualité de la prédiction sur le jeu de test ? Comment se compare-t-elle à la régression linéaire?


#REPONSE ECRITE (3 lignes maximum)

