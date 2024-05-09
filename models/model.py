from sklearn.ensemble import GradientBoostingClassifier
import os
from . import preprocessing


class Model:
    def __init__(self):
        # define os caminhos dos datasets
        self.caminho_dataset = os.path.join(os.path.dirname(
            os.path.dirname(__file__)), 'datasets', 'parkinsons.data')
        self.caminho_dataset_atualizado = os.path.join(os.path.dirname(
            os.path.dirname(__file__)), 'datasets', 'parkinsons_updrs.data')
        # separa em treino e teste
        self.X_treino = None
        self.X_teste = None
        self.y_treino = None
        self.y_teste = None
        # define o modelo
        self.modelo = GradientBoostingClassifier(
            learning_rate=0.1, n_estimators=50, max_depth=3)
        self.novos_dados = None
        self.previsao_final = None
        self.probabilidade_novos_dados = None

    def preprocess(self):
        # préprocessa os dados e os separa em treino e teste
        self.X_treino, self.X_teste, self.y_treino, self.y_teste = preprocessing.preprocess(
            self.caminho_dataset)

    def treinar_modelo(self):
        # treinar o modelo
        self.modelo.fit(self.X_treino, self.y_treino)

    def tratar_dataset(self):
        # criar um df com informações úteis no final mas inúteis para a IA
        self.novos_dados, self.previsao_final = preprocessing.preprocess_new_data(
            self.caminho_dataset_atualizado)

    def calcular_probabilidade(self):
        # calcular as probabilidades da pessoa ter parkinson pegando somente a classe 1 (ter parkinson)
        self.probabilidade_novos_dados = (self.modelo.predict_proba(self.novos_dados)[
            :, 1]*100).round(2)

        # criando uma coluna de probabilidades de ter parkinson
        self.previsao_final['probabilidade'] = self.probabilidade_novos_dados

    def salvar_dataset(self):
        # salvar o novo df
        self.previsao_final.to_csv('parkinsons_predict.csv', index=False)
