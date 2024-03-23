from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# Carregar base de dados
df = pd.read_csv('parkinsons.data')

X = df.drop(['status', 'name', 'APQ', 'D2',
            'Fhi(Hz)', 'Flo(Hz)', 'Fo(Hz)',
             'PPQ', 'RAP', 'spread1', 'spread2'], axis=1)

y = df['status']

X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2)

# Definir os classificadores e os seus parâmetros obs: poderia também ser usado uma Pipeline
classificadores = [
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    GradientBoostingClassifier()]
parametros = [
    # Parâmetros para DecisionTreeClassifier
    {'max_depth': [None, 5, 10, 20, 50]},
    # Parâmetros para RandomForestClassifier
    {'n_estimators': [50, 100, 200]},
    # Parâmetros para GradientBoostingClassifier
    {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.5]}
]
# Listas para armazenar os melhores resultados para cada classificador
melhores_parametros = []
melhor_pontuacao = []
teste_precisao = []

# Loop sobre os classificadores
for classificador, parametro in zip(classificadores, parametros):
    # Criar o objeto GridSearchCV
    grid_search = GridSearchCV(classificador, parametro, cv=5)

    # Ajustar o objeto GridSearchCV aos dados de treinamento
    grid_search.fit(X_treino, y_treino)

    # Melhores parâmetros encontrados
    melhores_parametros.append(grid_search.best_params_)

    # Melhor pontuação no conjunto de validação cruzada
    melhor_pontuacao.append(grid_search.best_score_)

    # Avaliar o desempenho no conjunto de teste
    teste_precisao.append(grid_search.score(X_teste, y_teste))


# Imprimir resultados para cada classificador
for i, classificador in enumerate(classificadores):
    print(f"\n  Classificador: {classificador.__class__.__name__}")
    print(f"Melhores parâmetros: {melhores_parametros[i]}")
    print(f"Melhor pontuação: {melhor_pontuacao[i]}")
    print(f"Acurácia no conjunto de teste: {teste_precisao[i]}")

print(f"\n  O melhor estimator foi: {grid_search.best_estimator_}")


'''
após diversos testes, 
é possível concluir que:
GradientBoostingClassifier(learning_rate=0.1, n_estimators=50)
foi o que obteve maior pontuação e acurácia nos testes.
 '''
