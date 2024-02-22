import streamlit as st
import pandas as pd
import csv
from modules import state_write

#Configuration des dimensions & affichage de la page
st.set_page_config(page_title="Import", 
                   page_icon=":mortar_board:", 
                   layout='wide')

liste_df =['DF Diabète', 'DF Vin', 'Autre DF selon import']
choix = st.sidebar.selectbox(
'Quel df voulez-vous traiter ?', liste_df
)

if choix == 'DF Diabète':
    df = pd.read_csv("SRC/diabete.csv")
    # df = pd.read_csv("C:/Users/murai/OneDrive/Documents/GitHub/FATAL_ML/SRC/diabete.csv")
    # if 'df' not in st.session_state:
    state_write(df)

elif choix == 'DF Vin':
    df = pd.read_csv("SRC/vin.csv")
    # df = pd.read_csv("C:/Users/murai/OneDrive/Documents/GitHub/FATAL_ML/SRC/vin.csv")
    # if 'df' not in st.session_state:
    state_write(df)

else:
    file = st.file_uploader("Uploader un fichier", type="csv")
    button = st.button('importer')
    if button:

        content = file.getvalue().decode("utf-8")
        
        # Utiliser csv.Sniffer pour identifier automatiquement le séparateur
        dialect = csv.Sniffer().sniff(content)
        separator = dialect.delimiter

        # Lire le fichier CSV dans un DataFrame pandas en utilisant le séparateur identifié
        df = pd.read_csv(file, sep=separator)
        state_write(df)


