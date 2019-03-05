##############################################################################
# Master 2 Econometrics & Statistics 
# Universit� Toulouse-Capitole 
# Ann�e 2 - Semestre 2 - Big Data 
# @author: Julnor Georges, Boris Ombede, Rose-Camille Vincent
##############################################################################



##############################################################################
#
#    ALGORITHMIE DU BIG DATA
#
##############################################################################


#
# QUESTION 0 - IMPORTATION DES PACKAGES ET LIBRAIRIES UTILISEES PAR LA SUITE
# 


    dev.off()
    rm(list=ls())
    
    library(plyr)
    library(RColorBrewer)
    library(biglm)
    library(DBI)
    library(RSQLite)
    library(biglm)
    library(ROCR)
    library(pscl)
    library(stats)
    library(data.table)
    library(bigtabulate)
    library(biganalytics)
    library(bigmemory)
    library(ff)
    library(tidyverse)


#
# QUESTION 1 - IMPORT DU JEU DE DONNEES
# 


### Q1.1 - Indiquer le dossier et le fichier cible

    setwd("C:/Users/rosec/Dropbox/TSE/BigData/Project/bigdataprojects/TP2")


### Q1.2 - Importer les jeux de données complets et échantillonnés

# ---------- Utiliser une librairie usuelle (version de fichier échantillonnée)
    
    train_echantillon <- fread("train_echantillon.csv")
    train_echantillon <- as.data.frame(train_echantillon)


# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory) (version complète du fichier)

    train <- read.big.matrix("train.csv", header=T, type = "integer")


#
# QUESTION 2 - PREPARATION/NETTOYAGE DU JEU DE DONNEES
# 

### Q2.1 - Nettoyer et préparer les données


# Enlever les valeurs incorrectes ou manquantes (si pertinent)


# ---------- Utiliser une librairie usuelle
    
    dim(train_echantillon)
    head(train_echantillon)
    summary(train_echantillon)
    # On remarque qu' il y a des valeurs manquantes pour les variables "dropoff_longitude et dropoff_latitude" ==> on les supprime
    # On remarque des valeurs negatives pour "fare_amount". On les supprime egalement 
    train_echantillon <- na.omit(train_echantillon)
    train_echantillon[!complete.cases(train_echantillon)]
    train_echantillon <- train_echantillon[which(train_echantillon$fare_amount>=0),]
    
    # on remarque aussi que pour certaines observations, le nombre de passagers dans le taxi depasse un nombre raisonnable (208)
    # On decide de retenir 6 passagers au maximum etant donne que 99% des taxis de lechantillon ont 6 passagers ou moins 
    quantile(train_echantillon$passenger_count, p=c(0.25, 0.50, 0.75, 0.99))
    train_echantillon <- train_echantillon[which(train_echantillon$passenger_count<=6),]
    summary(train_echantillon)
    
    

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)
    dim(train)
    summary(train) # les memes constats sont faits pour la base "train"
    colnames(train)
    train <- train[train[, "fare_amount"] >= 0, ]
    train <- train[train[, "passenger_count"]<=6, ]
    
    summary(train)
    train <- na.omit(train)
    summary(train)
    


# Ne garder que les variables de géolocalisation (pour le jeu de données en entrée) et
# la variable "fare_amount" pour la sortie


# ---------- Utiliser une librairie usuelle

    colnames(train_echantillon)
    train_echantillon <- train_echantillon[, c("fare_amount", "pickup_longitude",
                                               "pickup_latitude","dropoff_longitude",
                                               "dropoff_latitude")]
    

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)
    colnames(train)
    train <- train[, c("fare_amount", "pickup_longitude",
                       "pickup_latitude","dropoff_longitude",
                       "dropoff_latitude")]
    


# Obtenir les caractéristiques statistiques de base des variables d'entrée et de sortie
# (par exemple, min, moyenne, mdéiane, max) et filter les valeurs aberrantes


# ---------- Utiliser une librairie usuelle

    summary(train_echantillon)
    # On remarque des valeurs aberrantes pour les variables de latitude et de longitude. Logiquement, les valeurs de latitude sont comprises entre -90 et +90 
     # et les valeurs de longitude entre -180 et +180. Bien qu' il soit difficile d'imaginer un taxi transportant un passer de New York a Shanghai (Chine),
     # on a tres peu d'information sur les parcours. Par consequent, on retiendra les intervalles [-39, +41] pour la latitude et [-73,+75] pour la longitude. 
    
    train_echantillon <- subset(train_echantillon, pickup_longitude >= -73 & pickup_longitude <= 75 & 
                                  dropoff_longitude >= -73 & dropoff_longitude <= 75 &
                                  pickup_latitude >= -39 & pickup_latitude <= 41 & 
                                  dropoff_latitude >= -39 & dropoff_latitude <= 41)
    

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)
    
    summary(train) # le meme constant est fait concernant la plus grande base 
    train <- train[train[,"pickup_longitude"] >= -73 & train[ ,"pickup_longitude"] <= 75 &
                        train[,"dropoff_longitude"] >= -73 & train[ ,"dropoff_longitude"] <= 75 &
                        train[,"pickup_latitude"]>= -39 & train[ ,"pickup_latitude"] <= 41 &
                        train[,"dropoff_latitude"]>= -39 & train[ ,"dropoff_latitude"] <= 41, ]
  
# Visualiser les distributions des variables d'entrée et de sortie (histogramme, pairplot)
    summary(train_echantillon)
    summary(train)

# ---------- Utiliser une librairie usuelle
  
    regvar <- c("fare_amount", "fare_amount","pickup_longitude","pickup_latitude", "dropoff_longitude","dropoff_latitude" )
    train <- as.data.frame(scale(apply(train, 2, as.numeric)))
   
    #for (var in regvar){ hist(train[, var], main=var)} -> run later
    #pairs(train[, regvar]) -> run later




# Séparer la variable à prédire ("fare_amount") des autres variables d'entrée
# Créer un objet avec variables d'entrée et un objet avec valeurs de sortie (i.e. "fare_amount")



# ---------- Utiliser une librairie usuelle

    input_c <- train_echantillon[ , -grep("fare_amount", names(train_echantillon))]
    output_c <- train_echantillon[ , "fare_amount"]
    
# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

    input_b <- train[ , -grep("fare_amount", names(train))]
    output_b <- train[ , "fare_amount"]
    


# Standardiser la matrice d'entrée et les vecteurs de sortie (créer un nouvel objet)


# ---------- Utiliser une librairie usuelle

CODE


# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)


CODE







#
# QUESTION 3 - CLUSTERING DU JEU DE DONNEES
# 




### Q3.1 - Réaliser un clustering k-means sur les données d'entrée standardisées


# ---------- Utiliser une librairie usuelle

CODE

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

CODE



### Q3.2 - Tracer la figure de l'inertie intraclusters et du R² en fonction du nombre de  clusters


# ---------- Utiliser une librairie usuelle

CODE





### Q3.3 - A partir de combien de clusters on peut dire que partitionner n'apporte plus 
###        grand chose? Pourquoi?



REPONSE ECRITE (3 lignes maximum)





### Q3.4 - Comment pouvez-vous qualifier les clusters obtenus selon les variables originales?
###        Par exemple, y a-t-il des clusters selon la localisation ? 




REPONSE ECRITE (3 lignes maximum)



### Q3.5 - Visualiser les clusters avec des couleurs différentes sur un 'pairplot' avec plusieurs variables


# ---------- Utiliser une librairie usuelle


CODE









#
# QUESTION 4 - ANALYSE EN COMPOSANTES PRINCIPALES (ACP) POUR SIMPLIFIER LE JEU DE DONNEES
# 



### Q4.1 - Faire une ACP sur le jeu de données standardisé


# ---------- Utiliser une librairie usuelle

CODE

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

CODE


### Q4.2 - Réaliser le diagnostic de variance avec un graphique à barre (barchart)

 

# ---------- Utiliser une librairie usuelle


CODE




### Q4.3 - Combien de composantes doit-on garder? Pourquoi?
       


REPONSE ECRITE (3 lignes maximum)




### Q4.4 - Tracer un graphique 'biplot' indiquant les variables initiales selon les 2 premières CP
###        Sélectionner éventuellement un sous-échantillon de points pour faciliter la visualisation

 

# ---------- Utiliser une librairie usuelle


CODE




### Q4.5 - Comment les variables initiales se situent-elles par rapport aux 2 premières CP? 


REPONSE ECRITE (3 lignes maximum)











#
# QUESTION 5 - REGRESSION LINEAIRE
# 



### Q5.1 - Mener une régression linéaire de la sortie "fare_amount" 
###        en fonction de l'entrée (mise à l'échelle), sur tout le jeu de données


# ---------- Utiliser une librairie usuelle

CODE

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

CODE


### Q5.2 - Que pouvez-vous dire des résultats du modèle? Quelles variables sont significatives?



REPONSE ECRITE (3 lignes maximum)



### Q5.3 - Prédire le prix de la course en fonction de nouvelles entrées avec une régression linéaire


# Diviser le jeu de données initial en échantillons d'apprentissage (60% des données), validation (20%) et test (20%)


# ---------- Utiliser une librairie usuelle

CODE

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

CODE


# Réaliser la régression linéaire sur l'échantillon d'apprentissage, tester plusieurs valeurs
# de régularisation (hyperparamètre de la régression linéaire) et la qualité de prédiction sur l'échantillon de validation. 


# ---------- Utiliser une librairie usuelle

CODE

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

CODE

# Calculer le RMSE et le R² sur le jeu de test.



# ---------- Utiliser une librairie usuelle

CODE

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

CODE

# Quelle est la qualité de la prédiction sur le jeu de test ?


REPONSE ECRITE (3 lignes maximum)








#
# QUESTION 6 - REGRESSION LOGISTIQUE
# 



### Q6.1 - Mener une régression logisitique de la sortie "fare_amount" (après binarisation selon la médiane) 
###        en fonction de l'entrée (mise à l'échelle), sur tout le jeu de données


# Créer la sortie binaire 'fare_binaire' en prenant la valeur médiane de "fare_amount" comme seuil


# ---------- Utiliser une librairie usuelle

CODE

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

CODE


# Mener la régression logistique de "fare_binaire" en fonction des entrées standardisées


# ---------- Utiliser une librairie usuelle

CODE

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

CODE




### Q6.2 - Que pouvez-vous dire des résultats du modèle? Quelles variables sont significatives?



REPONSE ECRITE (3 lignes maximum)



### Q6.3 - Prédire la probabilité que la course soit plus élevée que la médiane
#           en fonction de nouvelles entrées avec une régression linéaire


# Diviser le jeu de données initial en échantillons d'apprentissage (60% des données), validation (20%) et test (20%)


# ---------- Utiliser une librairie usuelle

CODE

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

CODE


# Réaliser la régression logistique sur l'échantillon d'apprentissage et en testant plusieurs valeurs
# de régularisation (hyperparamètre de la régression logistique) sur l'échantillon de validation. 


# ---------- Utiliser une librairie usuelle

CODE

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

CODE


# Calculer la précision (accuracy) et l'AUC de la prédiction sur le jeu de test.



# ---------- Utiliser une librairie usuelle

CODE

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

CODE


# Quelle est la qualité de la prédiction sur le jeu de test ?


REPONSE ECRITE (3 lignes maximum)







#
# QUESTION 7 - RESEAU DE NEURONES (QUESTION BONUS)
# 



### Q7.1 - Mener une régression de la sortie "fare_amount" en fonction de l'entrée (mise à l'échelle), 
###       sur tout le jeu de données, avec un réseau à 2 couches cachées de 10 neurones chacune



# ---------- Utiliser une librairie usuelle

CODE

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

CODE



### Q7.2 - Prédire le prix de la course en fonction de nouvelles entrées avec le réseau de neurones entraîné


# Diviser le jeu de données initial en échantillons d'apprentissage (60% des données), validation (20%) et test (20%)


# ---------- Utiliser une librairie usuelle

CODE

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

CODE


# Réaliser la régression avec réseau de neurones sur l'échantillon d'apprentissage et en testant plusieurs 
# nombre de couches et de neurones par couche sur l'échantillon de validation. 


# ---------- Utiliser une librairie usuelle

CODE

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

CODE


# Calculer le RMSE et le R² de la meilleure prédiction sur le jeu de test.


# ---------- Utiliser une librairie usuelle

CODE

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

CODE

# Quelle est la qualité de la prédiction sur le jeu de test ? Comment se compare-t-elle à la régression linéaire?


REPONSE ECRITE (3 lignes maximum)


