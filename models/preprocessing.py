import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier


def preprocess(caminho_dataset):
    # carregar a base de dados e colocar na variável df
    df = pd.read_csv(caminho_dataset)

    # colocar as outras colunas para prever o y excluindo colunas que não tem no dataset final
    X = df.drop(['status', 'name', 'APQ', 'D2',
                'Fhi(Hz)', 'Flo(Hz)', 'Fo(Hz)',
                 'PPQ', 'RAP', 'spread1', 'spread2'], axis=1)

    # colocar a coluna a ser prevista em y
    y = df['status']

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.2)

    # definir os processos a serem feitos nos dados
    pipe = Pipeline(steps=[('StandardScaler', StandardScaler()),
                           ('MinMaxScaler', MinMaxScaler()),
                           ('GradientBoostingClassifier', GradientBoostingClassifier(learning_rate=0.1, n_estimators=50))])  # max_depth padrão foi o que convergiu melhor

    return pipe, X_treino, X_teste, y_treino, y_teste


def preprocess_new_data(caminho_novo_dataset):

    novos_dados = pd.read_csv(caminho_novo_dataset)

    # separando os dados subject e outros para a tabela final
    previsao_final = novos_dados[[
        'subject#', 'age', 'sex', 'test_time']]

    # tirar as informações não foram treinadas
    novos_dados = novos_dados.drop(
        ['subject#', 'age', 'sex', 'test_time', 'motor_UPDRS', 'total_UPDRS', 'Jitter:PPQ5',
            'Jitter:RAP', 'Shimmer:APQ11'], axis=1)

    return novos_dados, previsao_final
