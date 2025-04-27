import matplotlib.pyplot as plt
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import numpy as np;
import pandas as pd;
import math

dataset = pd.read_excel("nobel.xlsx")
dataset = dataset.iloc[:, 1:]


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
    total_row = pd.DataFrame([total_row], columns=dataset.columns)
    dataset = pd.concat([dataset, total_row], ignore_index=True)
    
    return dataset

Effectif = Add_Total(dataset)
print(Effectif)

def Frequences_Absolues(eff):
    freq = eff.copy()
    for i in range(len_rows):
        for j in range(len_columns+1):
            freq.iloc[i,j] = eff.iloc[i,j]/1533
    return freq

Frequences = Frequences_Absolues(dataset)
print(Frequences)
Freq = Frequences.astype(float)

def Profile_ligne(eff):
    freq = eff.copy()
    for i in range(0,len_rows):
        for j in range(1,len_columns):
            freq.iloc[i,j] = eff.iloc[i,j]/eff.iloc[i,len_columns]
    return freq
print(Profile_ligne(data))

def Profile_colonne(eff):
    freq = eff.copy() 
    for i in range(0,len_rows):   
        for j in range(1,len_columns):
            freq.iloc[i,j] = eff.iloc[i,j]/eff.iloc[len_rows,j]
    freq = Add_Total(freq.iloc[:len_rows-1, :len_columns])
    return freq