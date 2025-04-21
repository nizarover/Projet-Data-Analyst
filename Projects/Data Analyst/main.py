import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

pltShow = True

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
dataMatrix = pd.read_csv("C:\\Users\\exe\\Desktop\\Scripts\\Coding\\Python\\Projects\\Data Analyst\\diabetes.csv").to_numpy().T

print(pd.DataFrame(dataMatrix).T)
dataMatrix_CR = dataMatrix

# Suppression de certaines lignes spécifiques de la matrice
dataMatrix_CR = np.delete(dataMatrix_CR, 7, axis=0)  # Suppression de la ligne 7
dataMatrix_CR = np.delete(dataMatrix_CR, 0, axis=0)  # Suppression de la ligne 0

ShowGraph(dataMatrix_CR[0], dataMatrix_CR[1], "Comparaison Entre le taux de Glucose et le taux d'Insulin \n avant centrage et réduction", "Glucose", "Insulin")

# Normalisation des variables (centrage et réduction)
for variable in dataMatrix_CR:
    moyenne = np.average(variable)  # Calcul de la moyenne
    ecart_type = np.std(variable)  # Calcul de l'écart-type
    for i in range(len(variable)):
        variable[i] = (variable[i] - moyenne) / ecart_type  # Normalisation

ShowGraph(dataMatrix_CR[0], dataMatrix_CR[1], "Comparaison Entre le taux de Glucose et le taux d'Insulin \n après centrage et réduction", "Glucose", "Insulin")

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

# Tri par la deuxième colonne (valeur d'inertie expliquée)
inertie_explique.sort(key=lambda x: x[1], reverse=True)
print(inertie_explique)  # Affichage de l'inertie expliquée triée

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
v1 = np.linalg.eig(Mat_de_Correlation)[1][:, inertie_explique[0][0]]
v2 = np.linalg.eig(Mat_de_Correlation)[1][:, inertie_explique[1][0]]

# Projection des données sur les deux premiers axes principaux
vector1 = dataMatrix_CR.T @ v1
vector2 = dataMatrix_CR.T @ v2

# Affichage du graphique des deux premiers axes principaux
ShowGraph(vector1, vector2, "Nuage des individus", "F1(33.31%)", "F2(16.37%)")

eig = np.linalg.eig(Mat_de_Correlation)
landas = eig.eigenvalues
vectors = eig.eigenvectors

def matriceCorrelation() :
    cercle_correlation = []
    for i in range(len(dataMatrix_CR)):
        temp = []
        for j in range(len(dataMatrix_CR)):
            temp += [vectors[i][j] * math.sqrt(landas[j])]
        cercle_correlation += [temp]
    
    cercle_correlation = np.array(cercle_correlation, dtype=float)
    return cercle_correlation

cercle_correlation = matriceCorrelation()
print(cercle_correlation)
ShowMatrix(cercle_correlation)

def ShowCircleCorrelation() :
    fig, ax = plt.subplots()
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal', adjustable='box')
    ax.axhline(0, color='black', linewidth=1)  # Axe horizontal
    ax.axvline(0, color='black', linewidth=1)  # Axe vertical
    circle = plt.Circle((0, 0), 1, color='blue', fill=False, linestyle='-')  # Cercle de corrélation
    ax.add_artist(circle)
    plt.title("Cercle de corrélation")
    plt.xlabel("F1")
    plt.ylabel("F2")
    plt.grid()  # Grille
    for i in range(len(cercle_correlation)):
        x = cercle_correlation[i][0]
        y = cercle_correlation[i][1]
        plt.plot(x, y, 'o', color='blue')
        plt.text(x, y, str(i), fontsize=12, ha='right', va='bottom')
    plt.show()
ShowCircleCorrelation()

# Calcul de la qualité de représentation des individus
qualite_representation = []
def RepresentationQuality() :
    print(len(dataMatrix_CR), len(dataMatrix_CR[0]))
    for i in range(18):
        somme_carres = np.linalg.norm(dataMatrix_CR.T[i], 2)**2
        q1 = (vector1[i]**2) / somme_carres
        q2 = (vector2[i]**2) / somme_carres
        qualite_representation.append([q1,q2,q1+q2])
    return qualite_representation
ShowMatrix(RepresentationQuality())
# Affichage de la qualité de représentation
print("Qualité de représentation des individus :")
for i, qualite in enumerate(qualite_representation):
    print(f"Individu {i + 1}: F1 = {qualite[0]:.2f}, F2 = {qualite[1]:.2f}, somme = {qualite[0] + qualite[1]}")

