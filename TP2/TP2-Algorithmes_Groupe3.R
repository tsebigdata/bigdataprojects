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
  
    dossier <- "C:/Users/rosec/Dropbox/TSE/BigData/Project/bigdataprojects/TP2/"
    train <- "train.csv"
    train_echantillon <- "train_echantillon.csv"
    chemin_train <- paste0(dossier, train)
    chemin_train_e <- paste0(dossier,train_echantillon)


### Q1.2 - Importer les jeux de données complets et échantillonnés

# ---------- Utiliser une librairie usuelle (version de fichier échantillonnée)
    
    train_echantillon <- fread(chemin_train_e)
    train_echantillon <- as.data.frame(train_echantillon)


# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory) (version complète du fichier)

    train <- read.big.matrix(chemin_train, header=T, type = "integer")

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
    # On decide dutiliser le IQR pr filtrer le nombre de passager 
    
    cutoff_train_e <- quantile(train_echantillon$passenger_count, 0.75) +1.5*IQR(train_echantillon$passenger_count)
    outlier_train_e<- which(train_echantillon$passenger_count>cutoff_train_e)
    train_echantillon <- train_echantillon[-outlier_train_e,]
    summary(train_echantillon)
    
    

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)
    dim(train)
    summary(train) # les memes constats sont faits pour la base "train"
    colnames(train)
    train <- train[train[, "fare_amount"] >= 0, ]
    train <- train[train[, "passenger_count"]<=3, ]
    
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

    input_c<-scale(input_c)
    output_c <-scale(output_c)
    
# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

    input_b <- scale(input_b)
    output_b <- scale(output_b)
    

#
# QUESTION 3 - CLUSTERING DU JEU DE DONNEES
# 

### Q3.1 - Réaliser un clustering k-means sur les données d'entrée standardisées


# ---------- Utiliser une librairie usuelle

    set.seed(20)
    kmeans_clusters_c <- kmeans(input_c, centers = 6,
                              iter.max = 100, algorithm = "Lloyd")
    str(kmeans_clusters_c)
    

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)
    #library(biganalytics)
   # set.seed(20)
   # k_means_clusters_b <- bigkmeans(input_b, centers = 6,
                                    iter.max = 100, nstart = 1,
                                    dist = "euclid")


### Q3.2 - Tracer la figure de l'inertie intraclusters et du R² en fonction du nombre de  clusters


# ---------- Utiliser une librairie usuelle
    
    for(cluster_num in 1:10){
      kmeans_clusters_c <- kmeans(input_c, centers = cluster_num,
                                iter.max = 200, algorithm = "Lloyd")
      print(paste0(cluster_num ,
                   " clusters - Inertie: ",
                   kmeans_clusters_c$tot.withinss))}


### Q3.3 - A partir de combien de clusters on peut dire que partitionner n'apporte plus 
###        grand chose? Pourquoi?
  
    # A partir du 6eme cluster


### Q3.4 - Comment pouvez-vous qualifier les clusters obtenus selon les variables originales?
###        Par exemple, y a-t-il des clusters selon la localisation ? 


  

### Q3.5 - Visualiser les clusters avec des couleurs différentes sur un 'pairplot' avec plusieurs variables


# ---------- Utiliser une librairie usuelle

    
    index_plot_c <- sample(nrow(input_c), 1000)
    pairs(input_c[index_plot_c, ], col= kmeans_clusters_c$cluster[index_plot_c],pch=19)


#
# QUESTION 4 - ANALYSE EN COMPOSANTES PRINCIPALES (ACP) POUR SIMPLIFIER LE JEU DE DONNEES
# 


### Q4.1 - Faire une ACP sur le jeu de données standardisé


# ---------- Utiliser une librairie usuelle
    summary(train_echantillon) #pas de valeurs manquantes
    output_c_binaire <- rep(0, nrow(train_echantillon))
    output_c_binaire[output_c>median(output_c)] <- 1
    
    input_c <- scale(input_c)
    
    ACP_c <- prcomp(input_c, center = T, scale. = T)
    print(ACP_c)
    

# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

library(bigpca)


### Q4.2 - Réaliser le diagnostic de variance avec un graphique à barre (barchart)

 
# ---------- Utiliser une librairie usuelle
  summary(ACP_c)
  plot(ACP_c)


### Q4.3 - Combien de composantes doit-on garder? Pourquoi?
       
  # On peut garder  2 composantes et on aura ainsi plus de 92% des variances.  


### Q4.4 - Tracer un graphique 'biplot' indiquant les variables initiales selon les 2 premières CP
###        Sélectionner éventuellement un sous-échantillon de points pour faciliter la visualisation

 
# ---------- Utiliser une librairie usuelle


  biplot(ACP_c)



### Q4.5 - Comment les variables initiales se situent-elles par rapport aux 2 premières CP? 


    # Elles sont tres liees a la composante principale, un peu moins a la composante secondaire

  
  
#
# QUESTION 5 - REGRESSION LINEAIRE
# 


### Q5.1 - Mener une régression linéaire de la sortie "fare_amount" 
###        en fonction de l'entrée (mise à l'échelle), sur tout le jeu de données


# ---------- Utiliser une librairie usuelle

  formule_lm <- as.formula(fare_amount ~ . )
  Model_lm_c <- lm(formule_lm, data = train_echantillon)
  summary(Model_lm_c)
  
  
# ---------- Utiliser une librairie 'Big Data' (Dask ou bigmemory)

  

### Q5.2 - Que pouvez-vous dire des résultats du modèle? Quelles variables sont significatives?
    # Aucune des



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


