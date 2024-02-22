import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import warnings
import missingno as msno
from ydata_profiling import ProfileReport
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.model_selection import KFold
import statsmodels.api as sm
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import re
warnings.filterwarnings("ignore")

def load_data(file_path, delimiter, decimal):
    df = pd.read_csv(file_path, delimiter=delimiter, decimal=decimal)
    df.dropna(inplace=True)
    label_encoder = LabelEncoder()
    
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = label_encoder.fit_transform(df[column].astype(str))
    df=rename_columns(df)
    return df

def rename_columns(df):

    def replace_special_characters(col_name):
        return re.sub(r'\W+', '_', col_name).lower()
    df.columns = [replace_special_characters(col) for col in df.columns]
    
    return df
def select_columns_to_remove(df):
    column_names = df.columns.tolist()
    selected_columns = st.multiselect("Sélectionner les colonnes à supprimer", column_names)
    if st.button("Supprimer les colonnes sélectionnées"):
        df.drop(columns=selected_columns, inplace=True)
    return df

def select_columns_to_display(df):
    column_names = df.columns.tolist()
    selected_columns = st.multiselect("Sélectionner les colonnes à afficher", column_names)
    if st.button("Afficher les colonnes sélectionnées"):
        st.write(df[selected_columns])

def select_target_variable(df):
    column_names = df.columns.tolist()
    selected_y = st.selectbox("Choisir la cible", column_names)

    encoder = LabelEncoder()
    df[selected_y] = encoder.fit_transform(df[selected_y])
    st.write("La variable cible a été encodée avec succès.")

    return selected_y

def get_column_names(df, selected_y):
    column_names = df.columns.tolist()
    column_names.remove(selected_y)
    return column_names

def standardize_data(df):
    non_numeric_columns = [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col])]
    non_standardizable_columns = [col for col in non_numeric_columns if col not in ['votre_colonne_a_exclure']]
    
    if non_standardizable_columns:
        st.write("Colonnes qui ne sont pas numériques et ne peuvent pas être standardisées :")
        for col in non_standardizable_columns:
            st.write(col)
    
    standardizable_columns = [col for col in df.columns if col not in non_standardizable_columns]
    
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

def calculate_optimal_features(df):
    columns = df.columns.tolist()
    best_r2 = -1  
    best_features = []  
    selected_y = st.session_state['selected_y']
    feature_columns = [col for col in columns if col != selected_y]
    for L in range(1, len(feature_columns)+1):
        for subset in itertools.combinations(feature_columns, L):
            X_subset = df[list(subset)]
            y = df[selected_y]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_subset)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            if r2 > best_r2:
                best_r2 = r2
                best_features = subset
    st.write("Meilleur R2 Score:", best_r2)
    st.write("Meilleures caractéristiques:", best_features)

def select_target_type_reg(df,select_regression_type):
    types_regression_numeric = ["Ridge", "Lasso", "ElasticNet", "DecisionTreeRegressor", "RandomForestRegressor", "SVR"]
    types_regression_categorical = ["LogisticRegression", "DecisionTreeClassifier", "RandomForestClassifier", "SVC"]
    
    if select_regression_type:
        if select_regression_type=="Régression":
            selected_type_reg = st.selectbox("Choisir le type de régression", types_regression_numeric)
        else:
            selected_type_reg = st.selectbox("Choisir le type de régression", types_regression_categorical)
        
        return selected_type_reg
    else:
        st.error("Aucune variable cible sélectionnée.")
        return None
    
def select_regression_type():
    regression_types = ["Classification", "Régression"]
    selected_regression_type = st.selectbox("Choisir le type de régression", regression_types)
    return selected_regression_type


def train_type_regression(df, selected_y, selected_type_reg):
    if not selected_y:
        st.error("Aucune variable cible sélectionnée. Veuillez choisir une variable cible.")
        return

    column_names = df.columns.tolist()
    column_names.remove(selected_y)

    if not column_names:
        st.error("Aucune colonne sélectionnée pour les variables explicatives. Veuillez sélectionner au moins une colonne.")
        return
    
    X = df[column_names]
    y = df[selected_y]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if selected_type_reg == "Ridge":
        type_Reg = Ridge()
    elif selected_type_reg == "Lasso":
        type_Reg = Lasso()
    elif selected_type_reg == "ElasticNet":
        type_Reg = ElasticNet()
    elif selected_type_reg == "LogisticRegression":
        type_Reg = LogisticRegression()
    elif selected_type_reg == "DecisionTreeClassifier":
        type_Reg = DecisionTreeClassifier()
    elif selected_type_reg == "DecisionTreeRegressor":
        type_Reg = DecisionTreeRegressor()
    elif selected_type_reg == "RandomForestClassifier":
        type_Reg = RandomForestClassifier()
    elif selected_type_reg == "RandomForestRegressor":
        type_Reg = RandomForestRegressor() 
    elif selected_type_reg == "SVC":
        type_Reg = SVC()
    elif selected_type_reg == "SVR":
        type_Reg = SVR()
    else:
        st.error("Type de régression non reconnu.")
        return
    
    type_Reg.fit(X_train, y_train)
    y_pred = type_Reg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write(f"{selected_type_reg} MSE :", mse)
    st.write(f"{selected_type_reg} R2 Score :", r2)

    if selected_type_reg in ["LogisticRegression", "DecisionTreeClassifier", "RandomForestClassifier", "SVC"]:
        cm = confusion_matrix(y_test, y_pred) #MATRICE CONFUSION
        cr = classification_report(y_test, y_pred) #MATRICE CONFUSION
        st.write(f'\n -------------\\\ Matrice de confusion  ///-------------\n')
        st.write(cm[:15, :15])
                # Extraire les 10 premières et 10 dernières lignes du rapport de classification
        st.write(cr)



    st.write(f'\n -------------\\\ KFold  ///-------------\n')
    kf = KFold(n_splits=5, shuffle=True, random_state=2021)
    split = list(kf.split(X, y))
    
    liste_r2 = []
    liste_confusion_matrices = []
    for i, (train_index, test_index) in enumerate(split):
        st.write(f'\n ---------------- Fold {i+1} ------------\n')
        st.write(f" -------------- Training on {len(train_index)} samples-------------- ")
        st.write(f" -------------- Validation on {len(test_index)} samples-------------- ")
       
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
       
        data_train = df.loc[train_index]
        data_test = df.loc[test_index]

        equation = f"{selected_y} ~ {' + '.join(column_names)}"
        lm = smf.ols(formula=equation, data=df.iloc[train_index]).fit()
        y_hat = lm.predict(df.iloc[test_index][column_names])
   
        plt.scatter(data_test[selected_y].values,y_hat)
        plt.plot(np.arange(0,50))
        plt.show()

        plt.scatter(y_test, y_hat)
        plt.plot(np.arange(y_test.min(), y_test.max() + 1), np.arange(y_test.min(), y_test.max() + 1), color='red')  # Diagonale y=x
        plt.xlabel('Valeurs réelles')
        plt.ylabel('Prédictions')
        plt.title(f'Fold {i+1} - Valeurs réelles vs Prédictions')
        st.pyplot(plt)

        type_Reg.fit(X_train, y_train)
       
        y_pred = type_Reg.predict(X_test)
       
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.write(f" Fold {i+1} :  MSE {round(mse, 4)}")
        st.write(f" Fold {i+1} :  R2 {round(r2, 4)}")
       
        liste_r2.append(r2)

    st.write(f"//////  Moyenne des R2 : {np.mean(liste_r2)}")



if 'data_loaded' not in st.session_state:
    st.session_state['data_loaded'] = False
if 'df' not in st.session_state:
    st.session_state['df'] = None
if 'selected_y' not in st.session_state:
    st.session_state['selected_y'] = None

st.write("//////////////////////////////////////////////////////IMPORT//////////////////////////////////////////////////////")
st.write('💎💎💎💎💎💎💎💎💎💎💎💎💎💎💎💎💎💎💎💎💎💎💎💎💎💎💎💎💎💎💎💎')
file_path = st.text_input("Entrez votre chemin de fichier", value="", key="nom_fichier_input")
delimiter = st.text_input("Entrez votre délimiteur (par défaut : ,)", value=",", key="delimiter_input")
decimal = st.text_input("Entrez votre décimal (par défaut : .)", value=".", key="decimal_input")

if st.button("Charger les données"):
    try:
        st.session_state['df'] = load_data(file_path, delimiter, decimal)
        st.session_state['data_loaded'] = True
    except FileNotFoundError:
        st.error("Erreur : Impossible de trouver le fichier spécifié.")
    except Exception as e:
        st.error(f"Une erreur s'est produite lors du chargement des données : {str(e)}")

if st.session_state['df'] is not None and not st.session_state['df'].empty:
    st.write("//////////////////////////////////////////////////////SUPPRESSION//////////////////////////////////////////////////////")
    select_columns_to_remove(st.session_state['df'])
    st.write("//////////////////////////////////////////////////////AFFICHAGE//////////////////////////////////////////////////////")
    select_columns_to_display(st.session_state['df'])
    st.write("///////////////////////////////////////////////////STANDARDIZATION///////////////////////////////////////////////////")
    standardize_data(st.session_state['df'])
    st.write("//////////////////////////////////////////////////////REGRESSION//////////////////////////////////////////////////////")
    # st.write("🐕‍🦺")
    st.session_state['selected_y'] = select_target_variable(st.session_state['df'])
    train_linear_regression(st.session_state['df'], st.session_state['selected_y'])

    if st.button("Lancer calcul optimal"):
        calculate_optimal_features(st.session_state['df'])
        st.session_state['calculate_button_clicked'] = True

    st.session_state['selected_type_'] = select_regression_type()
    st.session_state['selected_type_reg'] = select_target_type_reg(st.session_state['df'],st.session_state['selected_type_'])
    train_type_regression(st.session_state['df'], st.session_state['selected_y'],st.session_state['selected_type_reg'])
else:
    st.error("Erreur : Aucune donnée disponible.")

