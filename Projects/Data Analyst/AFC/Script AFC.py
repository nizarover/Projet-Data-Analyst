import matplotlib.pyplot as plt
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import numpy as np;
import pandas as pd;
import math
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

dataset = pd.read_excel("C:\\Users\\exe\\Desktop\\Scripts\\Coding\\Python\\Projects\\Data Analyst\\AFC\\nobel.xlsx") # Load data

rows_index = dataset.iloc[:,0] # Nom des Lignes
print(rows_index)
columns_index = dataset.columns[1:] # Nom des colonnes
print(columns_index)
dataset = dataset.iloc[:,1:] # Remove first column
N = dataset.values.sum() # sum of all values
print(N)

def Frequence(dataset): # function : return Tableau des fréquence
    freq = dataset.copy()
    for i in range(dataset.shape[0]):
        for j in range(dataset.shape[1]):
            freq.iloc[i,j] = freq.iloc[i,j] / N
    return freq

Freq = Frequence(dataset) # Initialize Tableau des fréquence
print(Freq)

somme_ligne = Freq.sum(axis=1) # Initialize fréquences marginales f_i.
print(somme_ligne)
somme_colonne = Freq.sum(axis=0) # Initialize fréquences marginales f_.j
print(somme_colonne)


def PieChart(pd_data,labels): # function : Show PieChart
    plt.pie(pd_data, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.show()
    
PieChart(somme_ligne, rows_index) # Camembert des fréquence des lignes
PieChart(somme_colonne, columns_index) # Camembert des fréquence des colonnes

def Matrice_X2(Freq): # function : Matrice de contribution X2
    MatX2 = Freq.copy()
    for i in range(Freq.shape[0]):
        for j in range(Freq.shape[1]):
            eff_theorique = somme_ligne[i] * somme_colonne[j]
            MatX2.iloc[i,j] = (Freq.iloc[i,j] - eff_theorique)**2 / eff_theorique
    return MatX2

MatContribution = Matrice_X2(Freq)
print(MatContribution)

def statistique_X2(Freq): # function : Calculate Chi2
    somme = 0
    for i in range(dataset.shape[0]):
        for j in range(dataset.shape[1]):
            eff_theorique = somme_ligne[i] * somme_colonne[j]
            somme += (Freq.iloc[i,j] - eff_theorique)**2 / eff_theorique
    return somme

X2 = statistique_X2(Freq)
print(X2)

def profile_ligne(Freq): # function : Calculate profile ligne
    pl = Freq.copy()
    for i in range(Freq.shape[0]):
        for j in range(Freq.shape[1]):
            pl.iloc[i,j] = pl.iloc[i,j]/somme_ligne[i]
    return pl

def profile_colonne(Freq): # function : Calculate profile colonne
    pl = Freq.copy()
    for i in range(Freq.shape[0]):
        for j in range(Freq.shape[1]):
            pl.iloc[i,j] = pl.iloc[i,j]/somme_colonne[j]
    return pl

prof_ligne = profile_ligne(Freq) # Matrice profile ligne
print(prof_ligne)
prof_colonne = profile_colonne(Freq) # Matrice profile colonne
print(prof_colonne)

def barres_empilees(Freq, rows_index, columns_index): # function : Show graphes en barres empilées
    fig, ax = plt.subplots(figsize=(10, 6))
    left = np.zeros(Freq.shape[0])
    for col in range(Freq.shape[1]):
        ax.barh(rows_index, Freq.iloc[:, col], left=left, label=columns_index[col])
        left += Freq.iloc[:, col]
    ax.set_ylabel('Lignes')
    ax.set_xlabel('Fréquences')
    ax.set_title('Graphes en barres empilées (horizontal)')
    ax.legend(title='Colonnes', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

barres_empilees(prof_ligne, rows_index, columns_index) # graphes en barres empilées des lignes
barres_empilees(prof_colonne.T, columns_index, rows_index) # graphes en barres empilées des colonnes

def Matrice_Corrige(Freq): # function : Matrice corrigé
    MatCorr = Freq.copy()
    for i in range(Freq.shape[0]):
        for j in range(Freq.shape[1]):
            eff_theorique = somme_ligne[i] * somme_colonne[j]
            MatCorr.iloc[i,j] = Freq.iloc[i,j] - eff_theorique / math.sqrt(eff_theorique)
    return MatCorr

MatCorr = Matrice_Corrige(Freq) # Matrice Corrigé
print(MatCorr)

U, s, Vt = np.linalg.svd(MatCorr, full_matrices=False) # Calcule SVD
# U : vecteurs propres (lignes)
# s : valeurs singulières (racines carrées des valeurs propres)
# Vt : vecteurs propres (colonnes)


ValPropres = s**2 # Valeurs Propres dans l'ordre décroissant
print("Valeurs Propres : ", ValPropres)
ValPropres_Pourcentage = ValPropres/sum(ValPropres) * 100 # Valeurs Propres en Pourcentage
print("Valeurs Propres (Pourcentages) : ", ValPropres_Pourcentage)

# Calcule des Valeurs Propres Cumulées
ValPropres_Cumule = []
somme = 0
for val in ValPropres_Pourcentage:
    somme += val
    ValPropres_Cumule.append(somme)
print("Valeurs Propres (Cumulé) : ", ValPropres_Cumule)

# Coordonnées lignes (tranches d'âge)
coord_lignes = U * s

# Coordonnées colonnes (réseaux sociaux)
coord_colonnes = Vt.T * s

def Nuage_de_Points(xCord, yCord):
    plt.figure(figsize=(10,10))
    
# Lignes
    plt.scatter(xCord[:,0], xCord[:,1], color='blue', label="")
    for i, txt in enumerate(rows_index):
        plt.annotate(txt, (xCord[i,0], xCord[i,1]), color='blue')

    # Colonnes
    plt.scatter(yCord[:,0], yCord[:,1], color='red', marker='^', label="")
    for i, txt in enumerate(columns_index):
        plt.annotate(txt, (yCord[i,0], yCord[i,1]), color='red')

    # Axes
    plt.axhline(0, color='grey', linestyle='--')
    plt.axvline(0, color='grey', linestyle='--')

    plt.title('Plan factoriel AFC')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.grid()
    plt.show()
    
Nuage_de_Points(coord_lignes, coord_colonnes)

def Histogramme_ValPropres_Cumule(ValPropres_Cumule):
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(ValPropres_Cumule) + 1), ValPropres_Cumule, color='skyblue', edgecolor='black')
    plt.xlabel('Dimensions')
    plt.ylabel('Valeurs Propres Cumulées (%)')
    plt.title('Histogramme des Valeurs Propres Cumulées')
    plt.xticks(range(1, len(ValPropres_Cumule) + 1))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

Histogramme_ValPropres_Cumule(ValPropres_Cumule)


    