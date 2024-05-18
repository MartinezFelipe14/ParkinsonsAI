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
        self.pipe = None
        self.novos_dados = None
        self.previsao_final = None
        self.probabilidade_novos_dados = None
        self.resultado = None

    def preprocess(self):
        # préprocessa os dados e os separa em treino e teste
        self.pipe, self.X_treino, self.X_teste, self.y_treino, self.y_teste = preprocessing.preprocess(
            self.caminho_dataset)

    def treinar_modelo(self):
        # treinar o modelo
        self.pipe.fit(self.X_treino, self.y_treino)

    def tratar_dataset(self):
        # criar um df com informações úteis no final mas inúteis para a IA
        self.novos_dados, self.previsao_final = preprocessing.preprocess_new_data(
            self.caminho_dataset_atualizado)

    def calcular_probabilidade(self):
        # calcular as probabilidades da pessoa ter parkinson pegando somente a classe 1 (ter parkinson)
        self.probabilidade_novos_dados = (self.pipe.predict_proba(self.novos_dados)[
            :, 1]*100).round(2)

        # criando uma feature de probabilidades de ter parkinson
        self.previsao_final['probabilidade'] = self.probabilidade_novos_dados

    def calcular_resultado(self):
        # calcular e rotular como ter ou não parkinson
        self.resultado = self.pipe.predict(self.novos_dados)

        # criando uma feature para os rótulos
        self.previsao_final['resultado'] = self.resultado

    def salvar_dataset(self):
        # salvar o novo df
        self.previsao_final.to_csv('parkinsons_predict.csv', index=False)
