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
import dask
import dask.dataframe as dd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.preprocessing import Imputer
import dask.array as da


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
train_set_clean.where(train_set_clean['fare_amount']<0,0) 


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
X.describe()
y.describe()


# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

print("***** Statistiques de base : Big data*****")
X_d.describe().compute()
y_d.describe().compute


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
import dask_ml
from dask_ml.preprocessing import StandardScaler
X_d_scaled = StandardScaler().fit_transform(X_d)
y_d_scaled = preprocessing.scale(y_d)




#
# QUESTION 3 - CLUSTERING DU JEU DE DONNEES
# 




### Q3.1 - Réaliser un clustering k-means sur les données d'entrée standardisées


# ---------- Utiliser une librairie usuelle
for k in range(1,10):
    kmeans_model = KMeans(n_clusters = k, random_state =1).fit(X_scaled)
    labels = kmeans_model.labels_
    inertia = kmeans_model.inertia_
    print("nombre de clusters:" + str(k) + " - Inertie:" +str(inertia))

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

from dask_ml.cluster import KMeans
kmeans_dask = KMeans(n_clusters = 4)
kmeans_dask.fit_transform(X_d_scaled)
cluster=kmeans_dask.labels



### Q3.2 - Tracer la figure de l'inertie intraclusters et du R² en fonction du nombre de  clusters


# ---------- Utiliser une librairie usuelle
from scipy.spatial.distance import cdist
distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k, random_state =1).fit(X_scaled)
    kmeanModel.fit(X_scaled)
    distortions.append(sum(np.min(cdist(X_scaled, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X_scaled.shape[0])

# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()





### Q3.3 - A partir de combien de clusters on peut dire que partitionner n'apporte plus 
###        grand chose? Pourquoi?

print("Quand le nombre de clusters atteint le point optimal, ce qui revient à dire plus K augmente plus les centroïdes approchent les centroïdes des grappes")
print("Les améliorations vont diminuer, à un moment donné, créant ainsi la forme du coude")
print("Dans notre exemple, on peut constater à partir du nombre de clusters égal K =4, l'inertie montre une certaine stagnation ou une très faible variation à la baisse")





### Q3.4 - Comment pouvez-vous qualifier les clusters obtenus selon les variables originales?
###        Par exemple, y a-t-il des clusters selon la localisation ? 




#REPONSE ECRITE (3 lignes maximum)



### Q3.5 - Visualiser les clusters avec des couleurs différentes sur un 'pairplot' avec plusieurs variables


# ---------- Utiliser une librairie usuelle

data_cluster = set_ech_clean[["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"]]
data_cluster["cluster"] = labels
sample_index = np.random.randint(0, len(X_scaled), 1000)
sns.pairplot(data_cluster.loc[sample_index, :], hue = "cluster")
plt.show()








#
# QUESTION 4 - ANALYSE EN COMPOSANTES PRINCIPALES (ACP) POUR SIMPLIFIER LE JEU DE DONNEES
# 



### Q4.1 - Faire une ACP sur le jeu de données standardisé


# ---------- Utiliser une librairie usuelle

from sklearn.preprocessing import StandardScaler
#imp = Imputer(missing_values ='NaN', strategy ='mean', axis=0)
#set_ech_imp = set_ech.copy()
#imp.fit(set_ech_imp)
X_acp, y_acp = set_ech_clean[input_var], set_ech_clean["fare_amount"]
X_acp_scaled = StandardScaler().fit_transform(X_acp)
y_binaire = np.zeros(len(y_acp))
y_binaire[y_acp>y_acp.median()]=1

import random

echantillon_plot = np.random.randint(0,len(X_acp_scaled), 1000)
plot_dataframe = pd.DataFrame(data=np.column_stack((y_binaire, X_acp_scaled)), columns = variable_keep)
plot_dataframe["fare_amount"] = plot_dataframe["fare_amount"].astype('category')


import seaborn as sns
sns.pairplot(plot_dataframe.loc[echantillon_plot, ], hue = "fare_amount")
plt.show()

import numpy as np
from sklearn.decomposition import PCA
pca = PCA(n_components = 4)
pca_resultat = pca.fit_transform(X_acp_scaled)


# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

import numpy as np
import dask.array as da
from dask_ml.decomposition import PCA
X_d_scaled = np.array(X_d_scaled)
dX = da.from_array(X_d_scaled,chunks = X_d_scaled.shape)
pca = PCA(n_components = 4)
pca.fit(dX)

pca = PCA(n_components=4, svd_solver ='full')
pca.fit(X_d)

### Q4.2 - Réaliser le diagnostic de variance avec un graphique à barre (barchart)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)


# ---------- Utiliser une librairie usuelle
print(pca.explained_variance_ratio_)
print(pca.singular_values_)


### Q4.3 - Combien de composantes doit-on garder? Pourquoi?
       


#REPONSE ECRITE (3 lignes maximum)




### Q4.4 - Tracer un graphique 'biplot' indiquant les variables initiales selon les 2 premières CP
###        Sélectionner éventuellement un sous-échantillon de points pour faciliter la visualisation

 

# ---------- Utiliser une librairie usuelle
pca_nouveau = pd.DataFrame(data=np.column_stack((y_binaire, pca_resultat)),columns = ['fare_amount', 'PC1', 'PC2', 'PC3', 'PC4'])
pca_nouveau['fare_amount'] = pca_nouveau["fare_amount"].astype('category')
sns.pairplot(pca_nouveau.loc[echantillon_plot, ], hue ="fare_amount")


xvector = pca_resultat.components_[0] 
yvector = pca_resultat.components_[1]

xs = pca_resultat.transform(X_acp_scaled)[:,0] 
ys = pca_resultat.transform(X_acp_scaled)[:,1]

for i in range(len(xvector)):
    plt.arrow(0, 0, xvector[i]*max(xs), yvector[i]*max(ys),
              color='r', width=0.0005, head_width=0.0025)
    plt.text(xvector[i]*max(xs)*1.2, yvector[i]*max(ys)*1.2,
             list(X_acp_scaled.columns.values)[i], color='r')

for i in range(len(xs)):
    plt.plot(xs[i], ys[i], 'bo')
    plt.text(xs[i]*1.2, ys[i]*1.2, list(X_acp_scaled.index)[i], color='b')

plt.show()






### Q4.5 - Comment les variables initiales se situent-elles par rapport aux 2 premières CP? 


#REPONSE ECRITE (3 lignes maximum)











#
# QUESTION 5 - REGRESSION LINEAIRE
# 



### Q5.1 - Mener une régression linéaire de la sortie "fare_amount" 
###        en fonction de l'entrée (mise à l'échelle), sur tout le jeu de données


# ---------- Utiliser une librairie usuelle
from sklearn import linear_model
X_scaled = preprocessing.scale(X)
y_scaled = preprocessing.scale(y)
regr = linear_model.LinearRegression()
regr.fit(X_scaled, y_scaled)
regr.score(X_scaled, y_scaled)
# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)
import dask_ml
from dask_ml.preprocessing import StandardScaler
X_d_scaled = StandardScaler().fit_transform(X_d)
y_d_scaled = preprocessing.scale(y_d)

from dask_glm.estimators import LinearRegression
lr = LinearRegression()
lr.fit(X_d_scaled, y_d_scaled)


### Q5.2 - Que pouvez-vous dire des résultats du modèle? Quelles variables sont significatives?



#REPONSE ECRITE (3 lignes maximum)



### Q5.3 - Prédire le prix de la course en fonction de nouvelles entrées avec une régression linéaire


# Diviser le jeu de données initial en échantillons d'apprentissage (60% des données), validation (20%) et test (20%)


# ---------- Utiliser une librairie usuelle

prediction_biglm = lr.predict(X_d_scaled)

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


