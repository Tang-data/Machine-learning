# #réequilibrage du jeu de données en utilisant SMOTE à insérer après la déclaration des FEATURES et des TARGET et avant le train_test_split
from collections import Counter
from imblearn.over_sampling import SMOTE

def isbalanced_or_not(colonne_target) :
    distribution = colonne_target.value_counts()
    is_balanced = all(count >= len(colonne_target) / len(distribution) * 0.8 for count in distribution)
    if is_balanced:
        print("Votre jeu de données semble déjà équilibré")
    else:
        print("Votre jeu de données n'est pas équilibré")
        reequilibrage_SMOTE()

#réequilibrage des données si les classes sont deséquilibrées
def reequilibrage_SMOTE():
    print('Distribution d origine %s' % Counter(y))
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    print('Après réarrangement des tailles d échantillon %s' % Counter(y_res))
    return X_res, y_res
#si jamais il est possible de changer X_res et y_res en X y ce serait plsu simple pour appeler les régressions ensuite
#peut etre aura t on besoin de garder les X et y d'origine
