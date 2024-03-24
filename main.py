from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd
import os
# import matplotlib.pyplot as plt
# import seaborn as sns

# mostrar o caminho completo do arquivo
caminho_dataset = os.path.join(os.path.dirname(
    __file__), 'datasets', 'parkinsons.data')

# carregar a base de dados e colocar na variável df
df = pd.read_csv(caminho_dataset)

# colocar as outras colunas para prever o y
X = df.drop(['status', 'name', 'APQ', 'D2',
            'Fhi(Hz)', 'Flo(Hz)', 'Fo(Hz)',
             'PPQ', 'RAP', 'spread1', 'spread2'], axis=1)

# colocar a coluna a ser prevista em y
y = df['status']

# dividir as informações de x e y em treino e teste em uma proporção de (0.75) e (0.25)
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2)

# criar o modelo
modelo_principal = GradientBoostingClassifier(
    learning_rate=0.1, n_estimators=50)

# treinar o modelo
modelo_principal.fit(X_treino, y_treino)

'''
# fazer uma previsão do modelo a fins de teste
previsao_modelo_principal = modelo_principal.predict(X_teste)

# testar a previsão do modelo
print(previsao_modelo_principal)

# criar uma matriz de confusão
matriz_confusao = confusion_matrix(y_teste, previsao_modelo_principal)

# plotar um gráfico com os (TP, FP, TN, FN)
sns.heatmap(matriz_confusao, annot=True, fmt='d',
            cmap='Blues')
plt.xlabel('Previsão do Modelo')
plt.ylabel('Dados Reais')
plt.title('Matriz de Confusão')
plt.show()
'''

# mostra o caminho do arquivo
caminho_dataset_atualizado = os.path.join(
    os.path.dirname(__file__), 'datasets', 'parkinsons_updrs.data')

# carregar a nova base de dados
novos_dados = pd.read_csv(caminho_dataset_atualizado)

# criar um df com informações úteis no final mas inúteis para a IA
previsao_final = novos_dados[['subject#', 'age', 'sex', 'test_time']]

# tirar as informações não foram treinadas
novos_dados = novos_dados.drop(
    ['subject#', 'age', 'sex', 'test_time', 'motor_UPDRS', 'total_UPDRS', 'Jitter:PPQ5',
     'Jitter:RAP', 'Shimmer:APQ11'], axis=1)

# calcular as probabilidades da pessoa ter parkinson pegando somente a classe 1 (ter parkinson)
probabilidade_novos_dados = (modelo_principal.predict_proba(novos_dados)[
    :, 1]*100).round(2)

# criando uma coluna de probabilidades de ter parkinson
previsao_final['probabilidade'] = probabilidade_novos_dados

# salvar o novo df
previsao_final.to_csv('parkinsons_predict.csv', index=False)
