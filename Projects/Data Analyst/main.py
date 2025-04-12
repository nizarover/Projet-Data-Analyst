import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

pltShow = False

# Fonction pour afficher un graphique de dispersion
def ShowGraph(array1, array2, title, xlabel, ylabel):
    if pltShow :
        plt.scatter(array1, array2, marker='o', color='blue', label='Data')
        plt.axhline(0, color='black', linewidth=1)  # Axe horizontal
        plt.axvline(0, color='black', linewidth=1)  # Axe vertical
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

# Fonction pour afficher une matrice sous forme graphique
def ShowMatrix(array):
    if pltShow :
        fig, ax = plt.subplots()
        cax = ax.matshow(array, cmap='coolwarm')  # Affichage de la matrice avec une palette de couleurs
        for (i, j), val in np.ndenumerate(array):
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', color='black')  # Valeurs dans la matrice
        fig.colorbar(cax)  # Barre de couleur
        ax.set_xticks(range(len(array)))
        ax.set_yticks(range(len(array)))
        plt.show()



# Chargement des données depuis un fichier Excel
dataMatrix = pd.read_excel("C:\\Users\\exe\\Desktop\\Scripts\\Coding\\Python\\Projects\\Data Analyst\\autos_acp.xls").to_numpy().T
dataMatrix_CR = dataMatrix

# Suppression de certaines lignes spécifiques de la matrice
dataMatrix_CR = np.delete(dataMatrix_CR, 7, axis=0)  # Suppression de la ligne 7
dataMatrix_CR = np.delete(dataMatrix_CR, 0, axis=0)  # Suppression de la ligne 0

ShowGraph(dataMatrix_CR[0], dataMatrix_CR[1], "Comparaison Entre nombre de cycles et puissance d'une voiture \n avant centrage et réduction", "nombre de cycles", "puissance")

# Normalisation des variables (centrage et réduction)
for variable in dataMatrix_CR:
    moyenne = np.average(variable)  # Calcul de la moyenne
    ecart_type = np.std(variable)  # Calcul de l'écart-type
    for i in range(len(variable)):
        variable[i] = (variable[i] - moyenne) / ecart_type  # Normalisation

ShowGraph(dataMatrix_CR[0], dataMatrix_CR[1], "Comparaison Entre nombre de cycles et puissance d'une voiture \n après centrage et réduction", "nombre de cycles", "puissance")

# Calcul de la matrice de corrélation

Mat_de_Correlation = (dataMatrix_CR @ dataMatrix_CR.T) / (len(dataMatrix_CR[0]))
Mat_de_Correlation = Mat_de_Correlation.astype(float)  # Conversion en type numérique

ShowMatrix(Mat_de_Correlation)

# Calcul des valeurs propres
valPropres = np.linalg.eigvals(Mat_de_Correlation).real.astype(float).tolist()

# Association des indices aux valeurs propres
valpropre_indexed = []
for i in range(len(valPropres)):
    valpropre_indexed.append([i, valPropres[i]])

print(valpropre_indexed)  # Affichage des valeurs propres indexées

# Calcul de l'inertie expliquée
inertie_explique = [[index, val] for index, val in valpropre_indexed]
for i in range(len(inertie_explique)):
    inertie_explique[i][1] *= 100 / sum(valPropres)  # Conversion en pourcentage

print(inertie_explique)  # Affichage de l'inertie expliquée

# Calcul de l'inertie cumulée
inertie_cumule = [[index, val] for index, val in inertie_explique]
cumule = 0
for i in range(len(inertie_cumule)):
    inertie_cumule[i][1] += cumule
    cumule += inertie_explique[i][1]

print(inertie_cumule)  # Affichage de l'inertie cumulée

# Calcul des deux plus grandes valeurs propres
landa1 = max(valPropres)
valPropres.remove(landa1)
landa2 = max(valPropres)

# Calcul des vecteurs propres associés
v1 = np.linalg.eig(Mat_de_Correlation)[1][:, np.argmax(np.linalg.eigvals(Mat_de_Correlation))]
v2 = np.linalg.eig(Mat_de_Correlation)[1][:, np.argmax(np.linalg.eigvals(Mat_de_Correlation) == landa2)]

# Projection des données sur les deux premiers axes principaux
vector1 = dataMatrix_CR.T @ v1
vector2 = dataMatrix_CR.T @ v2

# Affichage du graphique des deux premiers axes principaux
ShowGraph(vector1, vector2, "Nuage des individus", "F1(68.54%)", "F2(16.67%)")

# cercle_correlation = []
# eig = np.linalg.eig(Mat_de_Correlation)
# landas = eig.eigenvalues
# vectors = eig.eigenvectors
# for i in range(len(dataMatrix_CR)):
#     for j in range(len(dataMatrix_CR)):
#         cercle_correlation += [vectors[i][j] * math.sqrt(landas[j])]
        
# ShowGraph(cercle_correlation, "Cercle de Corrélation",  "F1(68.54%)", "F2(16.67%)")

