import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from customlazy.models import LazyClassifier, LazyRegressor
from modules import standardisation

# Configuration des dimensions & affichage de la page
st.set_page_config(page_title="Standardisation / LazyPredict", 
                   page_icon=":mortar_board:", 
                   layout='wide')


df = st.session_state['df']
colonne_target = st.session_state['colonne_target']
type_model = st.session_state['type_model']
X = df.select_dtypes('number').drop(colonne_target, axis = 1)
y= df[colonne_target]

# Detection de la standardisation des données
standardisation(df, colonne_target)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2,random_state =42)

lazy_box = st.sidebar.checkbox('Classer automatiquement les modèles')
if lazy_box:
    if type_model == 'reg':
        #Regression
        reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None )
        models,predictions = reg.fit(X_train, X_test, y_train, y_test)
    else:
        #Classification
        clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
        models,predictions = clf.fit(X_train, X_test, y_train, y_test)

    models = models.reset_index(['Model'])


    #print(df_model_lazypredicted)
    list_models = ['Ridge','Lasso','LinearRegression','ElasticNet','DecisionTreeRegressor','RandomForestRegressor','SVR','DecisionTreeClassifier','RandomForestClassifier','SVC','LogisticRegression']
    df_models_dispos = pd.DataFrame()
    for i,item in models['Model'].items() : 
        if item in list_models : 
            df_models_dispos = df_models_dispos._append(models.loc[i])
    df_models_dispos.reset_index(inplace=True)
    df_models_dispos.drop(columns=['index'],inplace=True)
    df_models_dispos
    
    st.session_state['df_models_dispos'] = df_models_dispos