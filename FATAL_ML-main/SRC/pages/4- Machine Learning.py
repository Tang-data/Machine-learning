import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from customlazy.models import LazyClassifier, LazyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.model_selection import GridSearchCV
from modules import (
    standardisation,
    calculate_optimal_features,
    selection_colonnes
)



# Configuration des dimensions & affichage de la page
st.set_page_config(page_title="Machine Learning", 
                   page_icon=":mortar_board:", 
                   layout='wide')


df = st.session_state['df']
colonne_target = st.session_state['colonne_target']
type_model = st.session_state['type_model']
X = df.select_dtypes('number').drop(colonne_target, axis = 1)
y= df[colonne_target]

# # Detection de la standardisation des données
# standardisation(df, colonne_target)

# # Train Test Split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2,random_state =42)

# lazy_box = st.sidebar.checkbox('Classer automatiquement les modèles')
# if lazy_box:
#     if type_model == 'reg':
#         #Regression
#         reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None )
#         models,predictions = reg.fit(X_train, X_test, y_train, y_test)
#     else:
#         #Classification
#         clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
#         models,predictions = clf.fit(X_train, X_test, y_train, y_test)

#     models = models.reset_index(['Model'])


#     #print(df_model_lazypredicted)
#     list_models = ['Ridge','Lasso','LinearRegression','ElasticNet','DecisionTreeRegressor','RandomForestRegressor','SVR','DecisionTreeClassifier','RandomForestClassifier','SVC','LogisticRegression']
#     df_models_dispos = pd.DataFrame()
#     for i,item in models['Model'].items() : 
#         if item in list_models : 
#             df_models_dispos = df_models_dispos._append(models.loc[i])
#     df_models_dispos.reset_index(inplace=True)
#     df_models_dispos.drop(columns=['index'],inplace=True)
#     df_models_dispos
df_models_dispos = st.session_state['df_models_dispos']

model_box = st.sidebar.checkbox('Choisir le modèle')
if model_box:
    liste_modeles =df_models_dispos['Model'].to_list()
    selected_model_ML = st.sidebar.selectbox(
    'Quel modèle voulez-vous utiliser ?', liste_modeles
    )
    st.session_state['selected_model_ML'] = selected_model_ML
    if selected_model_ML == "Ridge":
        model_ML = Ridge
    elif selected_model_ML == "LinearRegression":
        model_ML = LinearRegression
    elif selected_model_ML == "Lasso":
        model_ML = Lasso
    elif selected_model_ML == "ElasticNet":
        model_ML = ElasticNet
    elif selected_model_ML == "LogisticRegression":
        model_ML = LogisticRegression
    elif selected_model_ML == "DecisionTreeClassifier":
        model_ML = DecisionTreeClassifier
    elif selected_model_ML == "DecisionTreeRegressor":
        model_ML = DecisionTreeRegressor
    elif selected_model_ML == "RandomForestClassifier":
        model_ML = RandomForestClassifier
    elif selected_model_ML == "RandomForestRegressor":
        model_ML = RandomForestRegressor 
    elif selected_model_ML == "SVC":
        model_ML = SVC
    elif selected_model_ML == "SVR":
        model_ML = SVR
    else:
        st.error("Type de régression non reconnu.")
    st.session_state['model_ML'] = model_ML
    
    



button = st.sidebar.checkbox('Proposition de colonnes pertinentes')
if button:
    best_columns = calculate_optimal_features(df, model_ML())

col_box = st.sidebar.checkbox('Choisir les colonnes')
if col_box:
    # Selection mannuelle des colonnes pour la regression
    selection_col = selection_colonnes(df, best_columns)
    # liste_colonnes = df.columns.tolist()
    # selection_col = st.multiselect("Sélectionnez les colonnes", liste_colonnes, default=best_columns)
    new_X = selection_col.copy()
    # Train Test Split
    new_X_train, new_X_test, y_train, y_test = train_test_split(new_X, y,test_size=.2,random_state =42)
    st.session_state['new_X'] = new_X
    st.session_state['y'] = y
    st.session_state['new_X_train'] = new_X_train
    st.session_state['new_X_test'] = new_X_test
    st.session_state['y_train'] = y_train
    st.session_state['y_test'] = y_test


param_box = st.sidebar.checkbox('Proposition des meilleurs hyperparamètres')
if param_box:

    hyperparam_reg = {
        'LinearRegression': {
            'alpha': [0.1, 1.0, 10.0]
        },
        'Ridge': {
            'alpha': [0.1, 1.0, 10.0]
        },
        'Lasso': {
            'alpha': [0.1, 1.0, 10.0],
            'selection': ['cyclic', 'random']
        },
        'ElasticNet': {
            'alpha': [0.1, 1.0, 10.0],
            'l1_ratio': [0.1, 0.5, 0.9]
        },
        'DecisionTreeRegressor': {
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'RandomForestRegressor': {
            'n_estimators': [50, 100, 200],
            'max_features': ['auto', 'sqrt'],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'SVR': {
            'C': [0.1, 1.0, 10.0],
            'epsilon': [0.01, 0.1, 0.2]
        }
    }
        


    hyperparam_class = {
        'LogisticRegression': {
            'penalty': ['l1', 'l2'],
            'C': [0.001, 0.01, 0.1, 1, 10],
            'solver': ['liblinear', 'saga']
        },
        'DecisionTreeClassifier': {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'RandomForestClassifier': {
            'n_estimators': [50, 100, 200],
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt']
        },
        'SVC': {
            'C': [0.1, 1.0, 10.0],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'gamma': ['scale', 'auto']
        }
    }

    search = GridSearchCV(
            model_ML(),
            [hyperparam_reg[selected_model_ML] if type_model=='reg' else hyperparam_class[selected_model_ML] if type_model=='class' else None]	
    )

    search.fit(new_X,y)
    col_1, col_2 = st.columns(2)
    with col_1:
        st.write(f'Meilleurs hyperparamètres pour votre modèle {selected_model_ML} : ', search.best_params_)
    with col_2:
        st.metric(label='Meilleur score de validation', value=round(search.best_score_,4))
        
col_box = st.sidebar.checkbox('Entraîner le modèle')
if col_box:
    model = model_ML()
    model.set_params(**search.best_params_)
    model.fit(new_X_train, y_train)
    st.write('Votre modèle est bien entraîné, retrouvez vos résultats sur la page suivante')
    st.session_state['model'] = model