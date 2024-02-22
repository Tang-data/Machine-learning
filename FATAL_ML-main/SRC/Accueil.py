import streamlit as st

#Configuration des dimensions & affichage de la page
st.set_page_config(page_title="Accueil", 
                   page_icon=":mortar_board:", 
                   layout='wide')

st.markdown("""
<iframe src="https://docs.google.com/presentation/d/1SentJ8bIOtOEOyz9kGu_1sdY4lMLTr1ZDBiuwP5VXgI/embed?start=false&loop=false&delayms=3000" width="1250" height="900"></iframe>
""", unsafe_allow_html=True)

