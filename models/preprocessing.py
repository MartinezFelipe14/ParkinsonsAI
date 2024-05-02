import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess(caminho_dataset):
    # carregar a base de dados e colocar na variável df
    df = pd.read_csv(caminho_dataset)

    # colocar as outras colunas para prever o y
    X = df.drop(['status', 'name', 'APQ', 'D2',
                'Fhi(Hz)', 'Flo(Hz)', 'Fo(Hz)',
                 'PPQ', 'RAP', 'spread1', 'spread2'], axis=1)

    # colocar a coluna a ser prevista em y
    y = df['status']

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.2)

    sc = StandardScaler()
    sc.fit(X_treino)
    X_treino = sc.transform(X_treino.values)
    X_teste = sc.transform(X_teste.values)

    return X_treino, X_teste, y_treino, y_teste


def preprocess_new_data(caminho_novo_dataset):

    novos_dados = pd.read_csv(caminho_novo_dataset)

    previsao_final = novos_dados[[
        'subject#', 'age', 'sex', 'test_time']]

    # tirar as informações não foram treinadas
    novos_dados = novos_dados.drop(
        ['subject#', 'age', 'sex', 'test_time', 'motor_UPDRS', 'total_UPDRS', 'Jitter:PPQ5',
            'Jitter:RAP', 'Shimmer:APQ11'], axis=1)

    sc = StandardScaler()
    sc.fit(novos_dados)
    novos_dados = sc.transform(novos_dados.values)

    return novos_dados, previsao_final
