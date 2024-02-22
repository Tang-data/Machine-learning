import streamlit as st
from modules import (
    unnamed_drop, 
    suppression_colonne_vide, 
    affichage_blanc, 
    suppression_blanc, 
    selection_target, 
    encodage, 
    colinearite
)


# Configuration des dimensions & affichage de la page
st.set_page_config(page_title="Traitement", 
                   page_icon=":mortar_board:", 
                   layout='wide')
 
df = st.session_state['df']

for colonne in df.columns:
    if colonne=='Unnamed: 0':
        # Suppression de la colonne Unnamed créée par pandas
        unnamed_drop(df)



import streamlit as st

tab1, tab2, tab3 = st.tabs(["Valeurs manquantes", "Colonne cible", "Corrélations"])
with tab1:
    col_1, col_2, = st.columns(2)
    with col_1:
        # Suppression des colonnes vides
        empty_col_box = st.checkbox('Supprimer les colonnes vides')
    with col_2:
        if empty_col_box:
            (suppression_colonne_vide(df))
    
    # Ligne de séparation
    st.write("***")
    
    col_1, col_2, = st.columns(2)
    with col_1:
        # Affichage des None
        na_box = st.checkbox('Affichage des valeurs manquantes')
        if na_box:
            (affichage_blanc(df))
    with col_2:
        dropna_box = st.checkbox('Supprimer les valeurs manquantes')
        if dropna_box:
            suppression_blanc(df)


with tab2:   
    if 'target' in df.columns:
        colonne_target = 'target'
        st.session_state['colonne_target'] = colonne_target
    else:
        colonne_target = df.columns[-1]
        
        
    target_box = st.checkbox('Changer la colonne cible')
    if target_box:
        colonne_target = selection_target(df)
        st.session_state['colonne_target'] = colonne_target
    st.write(f"La colonne cible est {colonne_target}")

    if isinstance(df[colonne_target][0], (int, float)):
        type_model = 'reg'
        st.session_state['type_model'] = type_model
        st.session_state['df'] = df
    else:
        type_model = 'class'    
        # Encodage pour les classifications
        encod_box = st.sidebar.checkbox('Encoder')
        if encod_box:
            encodage(df, colonne_target)
        st.session_state['type_model'] = type_model
        st.session_state['df'] = df
    
with tab3:
    # Affichage des colinarités
    # colin_box = st.checkbox('Affichage des colinéarités')
    # if colin_box:  
    colinearite(df)

    col_1, col_2, col_3 = st.columns([1,3,1])
    with col_2:
        st.dataframe(df, height=730)



# # Suppression des colonnes vides
# empty_col_box = st.sidebar.checkbox('Supprimer les colonnes vides')
# if empty_col_box:
#     (suppression_colonne_vide(df))

# # Affichage des None
# na_box = st.sidebar.checkbox('Affichage des valeurs manquantes')
# if na_box:
#     (affichage_blanc(df))
# dropna_box = st.sidebar.checkbox('Supprimer les valeurs manquantes')
# if dropna_box:
#     suppression_blanc(df)

    
# if 'target' in df.columns:
#     colonne_target = 'target'
#     st.session_state['colonne_target'] = colonne_target
# else:
#     colonne_target = df.columns[-1]
    
    
# target_box = st.sidebar.checkbox('Changer la colonne cible')
# if target_box:
#     colonne_target = selection_target(df)
#     st.session_state['colonne_target'] = colonne_target
# st.sidebar.write(f"La colonne cible est {colonne_target}")

# if isinstance(df[colonne_target][0], (int, float)):
#     type_model = 'reg'
#     st.session_state['type_model'] = type_model
#     st.session_state['df'] = df
# else:
#     type_model = 'class'    
#     # Encodage pour les classifications
#     encod_box = st.sidebar.checkbox('Encoder')
#     if encod_box:
#         encodage(df, colonne_target)
#     st.session_state['type_model'] = type_model
#     st.session_state['df'] = df
    
# # Affichage des colinarités
# colin_box = st.sidebar.checkbox('Affichage des colinéarités')
# if colin_box:  
#     colinearite(df)

# col_1, col_2, col_3 = st.columns([1,3,1])
# with col_2:
#     st.dataframe(df, height=730)
