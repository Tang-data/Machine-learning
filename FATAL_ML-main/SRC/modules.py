import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import itertools


### IMPORT
def state_write(df):
    st.session_state['df'] = df
    st.write(df)
    



### TRAITEMENTS

#suppression de la colonne 'Unnamed: 0' créee lors de l'initialisation du Dataframe
def unnamed_drop(df) :
    df.drop(columns = ['Unnamed: 0'], inplace=True)

# affichage des titres de colonnes et des valeurs vides dessous
def affichage_blanc(df) :
    valeur_manquantes = df.isna().sum().to_frame()
    valeur_manquantes = valeur_manquantes.rename({0:'Nombre de valeurs manquantes'}, axis='columns')
    st.write("Nombre de valeur manquantes par colonne : ")
    st.write(valeur_manquantes)

# suppression des colonnes vides
def suppression_colonne_vide(df) :
    empty_columns = df.columns[df.isna().all()].tolist()
    if empty_columns:
        df.dropna(how='all', axis=1, inplace=True)
        st.write(f'Vous venez de supprimer les colonnes {empty_columns } ' )
    else:
        st.write(f'Votre dataframe ne contient pas de colonnes vides') 

# suppression des lignes ou des cases sont vides
def suppression_blanc(df) :
    long_init_df = len(df)
    df.dropna(how='any', axis=0, inplace=True)
    st.write(f'Vous venez de supprimer { long_init_df - len(df)} lignes' )

# selection colonne target
def selection_target(df) :
    liste_colonnes = df.columns.tolist()
    colonne_target = st.selectbox("Selectionnez la colonne cible : ", liste_colonnes)
    return colonne_target

# encodage manuel pour les classifications
def encodage(df, colonne_target):
    unique_values_colonne_target = df[colonne_target].unique()
    x=0
    st.write('Les valeurs de votre colonne cible ont été remplacées' )
    for i in unique_values_colonne_target : 
        df.replace(to_replace = i, value =x, inplace=True)
        st.write(f'remplacement de : {i} par {x}' )
        x+=1

# afficher colinearités
def colinearite(df) :
    mask = np.triu(df.select_dtypes("number").corr())
    fig, ax = plt.subplots(figsize=(10, 10))
    cmap = sns.diverging_palette(15, 160, n=11, s=100)
    
    col_1, col_2, col_3 = st.columns([1,3,1])
    with col_2:
        sns.heatmap(
            df.select_dtypes("number").corr(),
            mask=mask,
            annot=True,
            cmap=cmap,
            center=0,
            vmin=-1,
            vmax=1,
            ax=ax
        )
        st.pyplot(fig)
    
# Affiche l'histogramme de distribution pour chaque colonne l'une après l'autre
def affichage_distribution_colonnes (df):
    plt.figure(figsize=(10, 6))

    for column in df.columns:
        plt.hist(df[column], bins=10, alpha=0.5, label=column)
        plt.xlabel('Valeur')
        plt.ylabel('Fréquence')
        plt.title('Distribution des colonnes numériques')
        plt.legend()
        plt.grid(True)
        plt.show()

def affichage_nombre_lignes_par_colonne(df):
    nombre_de_valeurs_par_colonne = df.count()
    st.write("Nombre total de valeurs par colonne :")
    st.write(nombre_de_valeurs_par_colonne)





### MACHINE LEARNING

# selection manuelle des colonnes
def selection_colonnes(df, best_columns) :
    liste_colonnes = df.columns.tolist()
    selection_col = st.sidebar.multiselect("Sélectionnez les colonnes", liste_colonnes, default=best_columns)
    return df[selection_col]

def standardisation(df, colonne_target):
    # définir un seuil de proximité de 0
    threshold = 0.1
    # tester si la deviation std et la moyenne sont proche de 0
    close_to_zero_std = (df.std().abs() < threshold).all()
    close_to_zero_mean = (df.mean().abs() < threshold).all()

    if close_to_zero_std and close_to_zero_mean:
        st.write("Vos données semblent déjà standardisées")
    else:
        st.write("Vos données ne semblent pas standardisées")
        standard_box = st.sidebar.checkbox('Standardiser')
        if standard_box:
            standardize_data(df, colonne_target)

def standardize_data(df, colonne_target):
    non_numeric_columns = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
    non_standardizable_columns = [col for col in non_numeric_columns if col not in colonne_target]
    
    if non_standardizable_columns:
        st.write("Colonnes qui ne sont pas numériques et ne peuvent pas être standardisées :")
        for col in non_standardizable_columns:
            st.write(col)
    
    standardizable_columns = [col for col in df.columns if col not in [non_standardizable_columns, colonne_target]]
    
    if standardizable_columns:
        scaler = StandardScaler()
        df[standardizable_columns] = scaler.fit_transform(df[standardizable_columns])
        st.write("Les colonnes standardisables ont été standardisées avec succès.")
        st.session_state['standardized_data'] = df
    else:
        st.write("Aucune colonne standardisable n'a été trouvée.")

def train_linear_regression(df, selected_y):
    column_names = df.columns.tolist() 
    column_names.remove(selected_y)  
    selected_columns = st.multiselect("Sélectionner les colonnes à afficher", column_names,default=column_names)
    if not selected_columns:
        st.error("Aucune colonne sélectionnée. Veuillez choisir au moins une colonne.")
        return
    X = df[selected_columns]  
    y = df[selected_y]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write("MSE :", mse)
    st.write("R2 Score :", r2)

def calculate_optimal_features(df, type_model):
    columns = df.columns.tolist()
    best_r2 = -1  
    best_features = []  
    selected_y = st.session_state['colonne_target']
    feature_columns = [col for col in columns if col != selected_y]
    for L in range(1, len(feature_columns)+1):
        for subset in itertools.combinations(feature_columns, L):
            X_subset = df[list(subset)]
            y = df[selected_y]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_subset)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
            type_model.fit(X_train, y_train)
            y_pred = type_model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            if r2 > best_r2:
                best_r2 = r2
                best_features = subset
    st.write("Meilleur R2 Score:", round(best_r2,4))
    st.write("Meilleures caractéristiques:", best_features)
    best_columns = list(best_features)
    return best_columns
