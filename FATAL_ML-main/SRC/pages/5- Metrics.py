import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import statsmodels.formula.api as smf
import pickle

# Configuration des dimensions & affichage de la page
st.set_page_config(page_title="Metrics", 
                   page_icon=":chart_with_upwards_trend:", 
                   layout='wide')

df = st.session_state['df']
type_model = st.session_state['type_model']
model = st.session_state['model']
new_X = st.session_state['new_X']
y = st.session_state['y']
selected_model_ML = st.session_state['selected_model_ML']
new_X_train = st.session_state['new_X_train']
new_X_test = st.session_state['new_X_test']
y_train = st.session_state['y_train']
y_test = st.session_state['y_test']
y_pred = model.predict(new_X_test)

if type_model == 'reg' :
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    col_1, col_2 = st.columns(2)
    col_1.metric(label=f"{selected_model_ML} MSE :", value=round(mse,4))
    col_2.metric(label=f"{selected_model_ML} R2 Score :", value=round(r2,4))

if type_model =='class':
    cm = confusion_matrix(y_test, y_pred) #MATRICE CONFUSION
    cr = classification_report(y_test, y_pred, output_dict=True) # classification_report
    st.title(" Matrice de confusion : ")
    cm_df = pd.DataFrame(cm)
    st.dataframe(cm_df)
    st.title(" Rapport de classification : ")
    st.write(pd.DataFrame(cr).T.drop(index='accuracy'))




# Ligne de séparation
st.write("***")

if type_model == 'reg' :
    # st.write(f'\n -------------\\\ KFold  ///-------------\n')
    kf = KFold(n_splits=5, shuffle=True, random_state=2021)
    split = list(kf.split(new_X, y))
    column_names = df.columns.to_list()
    liste_r2 = []
    liste_confusion_matrices = []
    # Boucle sur les plis de validation croisée
    for i, (train_index, test_index) in enumerate(split):
        st.title(f'\n ------------ Fold {i+1} ------------\n')
        st.write(f" -------------- Training on {len(train_index)} samples -------------- ")
        st.write(f" -------------- Validation on {len(test_index)} samples -------------- ")
        
        # Séparation des données en ensembles d'entraînement et de test
        X_train, X_test = new_X.iloc[train_index], new_X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        data_train = df.loc[train_index]
        data_test = df.loc[test_index]
        # Création du modèle OLS
        equation = f"{y.name} ~ {' + '.join(column_names)}"
        lm = smf.ols(formula=equation, data=data_train).fit()
        
        # Entraînement du modèle sur l'ensemble d'entraînement et évaluation sur l'ensemble de test
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        plt.scatter(y_test, y_pred)
        plt.plot(np.arange(y_test.min(), y_test.max() + 1), np.arange(y_test.min(), y_test.max() + 1), color='red')
        plt.xlabel('Valeurs réelles')
        plt.ylabel('Prédictions')
        plt.title(f'Fold {i+1} - Valeurs réelles vs Prédictions')
        st.pyplot(plt)

        st.markdown(f"## Fold {i+1}")
        col_1, col_2 = st.columns(2)
        col_1.metric(label="MSE :", value=round(mse,4))
        col_2.metric(label="R2 Score :", value=round(r2,4))

        # Ligne de séparation
        st.write("***")
        
        liste_r2.append(r2)

    st.metric(label="  Moyenne des R2 : ", value=round(np.mean(liste_r2), 4))

save_model = st.sidebar.button('Sauvegardez votre modèle')
if save_model:
    #sauvergarde du modèle pour utilisation ultérieure 
    with open('modele.pkl', 'wb') as fichier_modele:
        pickle.dump(model, fichier_modele)


with open("SRC/modele.pkl", "rb") as file:
        model_bytes = file.read()
        st.sidebar.download_button(
        "Téléchargez votre modèle",
        model_bytes,
        "Model.pkl",
        mime="application/octet-stream",
        key='download-pkl'
        )