import matplotlib.pyplot as plt
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import numpy as np;
import pandas as pd;
import math

dataset = pd.read_excel("C:\\Users\\achraf\\Desktop\\Projet-Data-Analyst\\Projects\\FCA\\nobel.xlsx")


rows = dataset.iloc[:,0]
columns = dataset.columns[1:]

len_rows = len(rows)
len_columns = len(columns)

def Add_Total(dataset):
    # Calculate the total of each row
    dataset["Total"] = dataset[columns].sum(axis=1)
    
    # Calculate the total of each column
    total_row = dataset[columns].sum(axis=0)
    total_row["Total"] = dataset["Total"].sum()

    
    # Append the total row to the dataset
    total_row = total_row.to_frame().T
    dataset = pd.concat([dataset, total_row], ignore_index=True)
    
    return dataset

Effectif = Add_Total(dataset)
print(Effectif)
Effectif = Effectif.iloc[:, 1:]

def Frequences_Absolues(eff):
    freq = eff.copy()
    for i in range(len_rows):
        for j in range(len_columns):
            freq.iloc[i,j] = eff.iloc[i,j]/1533
    return freq

Frequences = Frequences_Absolues(Effectif)
Frequences = Add_Total(Frequences)
print(Frequences)


def Profile_ligne(eff):
    freq = eff.copy()
    for i in range(len_rows):
        for j in range(len_columns):
            freq.iloc[i,j] = eff.iloc[i,j]/eff.iloc[i,len_columns]
    return freq
print(Profile_ligne(Effectif))

def Profile_colonne(eff):
    freq = eff.copy() 
    for i in range(0,len_rows):   
        for j in range(len_columns):
            freq.iloc[i,j] = eff.iloc[i,j]/eff.iloc[len_rows,j]
    return freq
print(Profile_colonne(Effectif))

def Correspondance(eff):
    freq = eff.copy()
    for i in range(len_rows):
        for j in range(len_columns):
            freq.iloc[i,j] = (eff.iloc[i,j] - (eff.iloc[i,len_columns] * eff.iloc[len_rows,j] / eff.iloc[len_rows,len_columns])) / math.sqrt(eff.iloc[i,len_columns] * eff.iloc[len_rows,j] / eff.iloc[len_rows,len_columns])
    return freq

print(Correspondance(Effectif))

def Chi2(eff):
    somme = 0
    N = eff.iloc[len_rows,len_columns]
    for i in range(len_rows):
        for j in range(1,len_columns):
            eff_theorique = eff.iloc[i,len_columns] * eff.iloc[len_rows,j]/N
            somme += math.pow(eff.iloc[i,j] - eff_theorique, 2)/eff_theorique
    return somme
print(Chi2(Effectif))

# On reconstruit S : matrice des résidus standardisés
S_df = Correspondance(Effectif)

# On prend les lignes 0..len_rows-1 (les len_rows lignes d'origine)
# et les colonnes 0..len_columns-1 (les len_columns colonnes d'origine)
S = S_df.iloc[:len_rows, :len_columns].values

# Calcul des valeurs et vecteurs propres de Sᵀ S
M = S.T @ S
valeurs_propres, vecteurs_propres = np.linalg.eig(M)

# Tri décroissant
idx = np.argsort(valeurs_propres)[::-1]
valeurs_propres = valeurs_propres[idx]
vecteurs_propres = vecteurs_propres[:, idx]

print("Valeurs propres :", valeurs_propres)
print("Vecteurs propres :\n", vecteurs_propres)

# Prendre les k premiers vecteurs propres
k = 2  # Exemple : projections sur les 2 premiers axes factoriels

# Calcul des coordonnées des lignes (projections sur les axes factoriels)
coord_lignes = S @ vecteurs_propres[:, :k]
# Si on veut afficher les coordonnées des lignes
print("Coordonnées des lignes :\n", coord_lignes)

# Calcul des coordonnées des colonnes (projections sur les axes factoriels)
coord_colonnes = vecteurs_propres[:, :k] * np.sqrt(valeurs_propres[:k])[None, :]  # (J, k)
# Si on veut afficher les coordonnées des colonnes
print("Coordonnées des colonnes :\n", coord_colonnes)

# coord_lignes : matrice (I×k)
# coord_colonnes : matrice (J×k)
# valeurs_propres : vecteur (k,)

# 1. Cos² pour les lignes
cos2_lignes = (coord_lignes**2) / np.sum(coord_lignes**2, axis=1)[:, None]
print("Cos² pour les lignes :\n", cos2_lignes)
# 2. Contribution des lignes
contrib_lignes = (coord_lignes**2) / valeurs_propres[None, :]
print("Contribution des lignes :\n", contrib_lignes)

# 3. Cos² pour les colonnes
cos2_colonnes = (coord_colonnes**2) / np.sum(coord_colonnes**2, axis=1)[:, None]
print("Cos² pour les colonnes :\n", cos2_colonnes)

# 4. Contribution des colonnes
contrib_colonnes = (coord_colonnes**2) / valeurs_propres[None, :]
print("Contribution des colonnes :\n", contrib_colonnes)

seuil = 0.5  
# Somme des cos² sur les deux premiers axes
qualite_ligne = cos2_lignes[:, 0] + cos2_lignes[:, 1]  
# Index des lignes bien représentées
idx_bonnes = np.where(qualite_ligne >= seuil)[0]

# On ne conserve que ces lignes pour tracer
coord_lignes_filt = coord_lignes[idx_bonnes]
labels_lignes_filt = [rows[i] for i in idx_bonnes]
print("Coordonnées des lignes filtrées :\n", coord_lignes_filt)
print("Labels des lignes filtrées :\n", labels_lignes_filt)