import matplotlib.pyplot as plt
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import numpy as np;
import pandas as pd;
import math

dataset = pd.read_excel("C:\\Users\\exe\\Desktop\\Scripts\\Coding\\Python\\Projects\\Data Analyst\\AFC\\nobel.xlsx")

rows = dataset.iloc[:,0]
columns = dataset.columns[1:]

len_rows = dataset.shape[0]
len_columns = dataset.shape[1]

def Add_Total(dataset):
    # Calculate the total of each row
    dataset["Total"] = dataset[columns].sum(axis=1)
    
    # Calculate the total of each column
    total_row = dataset[columns].sum(axis=0)
    total_row["Total"] = dataset["Total"].sum()
    
    # Append the total row to the dataset
    total_row = pd.DataFrame([total_row], columns=dataset.columns)
    dataset = pd.concat([dataset, total_row], ignore_index=True)
    
    return dataset
Effectifs = Add_Total(dataset)

Effectifs_CR = Effectifs.copy()
Effectifs_CR = Effectifs_CR.iloc[:len_rows,1:len_columns]
# Center and reduce the contingency table
for i in range(len_rows):
    for j in range(1, len_columns):
        Effectifs_CR.iloc[i, j] = (Effectifs.iloc[i, j] - (Effectifs.iloc[i, len_columns] * Effectifs.iloc[len_rows, j] / Effectifs.iloc[len_rows, len_columns])) / math.sqrt(Effectifs.iloc[i, len_columns] * Effectifs.iloc[len_rows, j] / Effectifs.iloc[len_rows, len_columns])

print(Effectifs_CR)

def AfficherMatrice(array, x_labels, y_labels):
    """Afficher une matrice sous forme de carte thermique avec annotations."""
    fig, ax = plt.subplots()
    cax = ax.matshow(array, cmap='coolwarm')
    for (i, j), val in np.ndenumerate(array):
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='black')
    fig.colorbar(cax)
    ax.set_xticks(range(len(x_labels)))
    ax.set_yticks(range(len(y_labels)))
    ax.set_xticklabels(x_labels, rotation=90)
    ax.set_yticklabels(y_labels)
    plt.show()
Mat_de_Correlation = (Effectifs_CR @ Effectifs_CR.T) / len_columns
AfficherMatrice(Mat_de_Correlation, "", "")
print(Mat_de_Correlation)

valpropres = np.linalg.eigvals(Mat_de_Correlation).real.astype(float).tolist()
print(valpropres)

print(Effectifs_CR)


print("Eigenvalues (from SVD):")
# print(eigenvalues)
    

def Frequences_Absolues(eff):
    freq = eff.copy()
    for i in range(0,len_rows):
        for j in range(1,len_columns):
            freq.iloc[i,j] = eff.iloc[i,j]/1533
    freq = Add_Total(freq.iloc[:len_rows-1, :len_columns])
    return freq
Frequences = Frequences_Absolues(dataset)

def Profile_ligne(eff):
    freq = eff.copy()
    for i in range(0,len_rows):
        for j in range(1,len_columns):
            freq.iloc[i,j] = eff.iloc[i,j]/eff.iloc[i,len_columns]
    freq = Add_Total(freq.iloc[:len_rows-1, :len_columns])
    return freq
def Profile_colonne(eff):
    freq = eff.copy() 
    for i in range(0,len_rows):   
        for j in range(1,len_columns):
            freq.iloc[i,j] = eff.iloc[i,j]/eff.iloc[len_rows,j]
    freq = Add_Total(freq.iloc[:len_rows-1, :len_columns])
    return freq

prof_ligne = Profile_ligne(Effectifs)
prof_colonne = Profile_colonne(Effectifs)

print(prof_ligne)
print(prof_colonne)

def statistique_X2(eff):
    somme = 0
    N = eff.iloc[len_rows,len_columns]
    for i in range(len_rows):
        for j in range(1,len_columns):
            eff_theorique = eff.iloc[i,len_columns] * eff.iloc[len_rows,j]/N
            somme += math.pow(eff.iloc[i,j] - eff_theorique, 2)/eff_theorique
    return somme

def Distance_X2_Centre(freq, i):
    somme = 0
    for j in range(1,len_columns):
        somme += ((freq.iloc[i,j]/freq.iloc[i,len_columns]) - freq.iloc[len_rows-1, j])**2 / freq.iloc[len_rows-1, j]    
    return somme 

def Distance_X2(freq, i, i_): 
    somme = 0
    for j in range(1,len_columns):
        somme += ((freq.iloc[i,j]/freq.iloc[i,len_columns]) - (freq.iloc[i_, j]/freq.iloc[i_, len_columns]))**2 / freq.iloc[len_rows-1, j]    
    return somme 
    
def Matrice_des_distances(freq):
    Mat = []
    for i in range(len_rows):
        subList = []
        for i_ in range(len_rows):
            dis = Distance_X2(freq, i, i_)
            subList.append(dis)
        Mat.append(subList)
    return [[float(value) for value in row] for row in Mat]

# Assuming 'dist_matrix' is your Chi-square distance matrix
dist_matrix = Matrice_des_distances(Frequences)  # A square distance matrix

D = Matrice_des_distances(Frequences)

def AfficherMatrice(array, x_labels, y_labels):
    """Afficher une matrice sous forme de carte thermique avec annotations."""
    fig, ax = plt.subplots()
    cax = ax.matshow(array, cmap='coolwarm')
    for (i, j), val in np.ndenumerate(array):
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='black')
    fig.colorbar(cax)
    ax.set_xticks(range(len(x_labels)))
    ax.set_yticks(range(len(y_labels)))
    ax.set_xticklabels(x_labels, rotation=90)
    ax.set_yticklabels(y_labels)
    plt.show()
    

