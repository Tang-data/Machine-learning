import streamlit as st
from PIL import Image

# Configuration des dimensions & affichage de la page
st.set_page_config(page_title="Metrics", 
                   page_icon=":chart_with_upwards_trend:", 
                   layout='wide',
                   initial_sidebar_state='collapsed')

#Importation image accueil
# image = Image.open('merci.gif')
# st.image(image)

# Initialisation du fond d'écran
import base64
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('SRC/merci.gif')



# TITRE DEFILANT
# Définition du texte à afficher
texte = "Merci pour votre attention"
# Configuration de la vitesse de défilement (en pixels par seconde)
vitesse = 20
# Génération du code HTML pour le texte défilant
html_code = f"""<marquee scrollamount="{vitesse}"> {texte} </marquee>"""
# Mise en forme taille = 150 pixels
st.markdown("""<style>
        .big-font {
            font-size:150px
        }
        </style>""", unsafe_allow_html=True)
# Affichage du texte défilant
st.markdown(f'<br><br><br><br><p class="big-font"><font face="Brush Script MT">{html_code}</font></p>', unsafe_allow_html=True)