from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def view_model(pipe, X_teste, y_teste, matrix=True, scatter=True):
    # fazer uma previsão do modelo a fins de teste
    previsao_modelo_principal = pipe.predict(X_teste)
    if matrix:
        # criar uma matriz de confusão
        matriz_confusao = confusion_matrix(
            y_teste, previsao_modelo_principal)

        # plotar um gráfico com os (TP, FP, TN, FN)
        sns.heatmap(matriz_confusao, annot=True, fmt='d',
                    cmap='Blues')
        plt.xlabel('Previsão do Modelo')
        plt.ylabel('Dados Reais')
        plt.title('Matriz de Confusão')
        plt.show()
    if scatter:
        df_resultados = pd.DataFrame({'Paciente': range(len(y_teste)),
                                      'Dados Reais': y_teste,
                                      'Previsão do Modelo': previsao_modelo_principal})
        # Plotar o gráfico de dispersão
        plt.scatter(
            df_resultados['Paciente'], df_resultados['Dados Reais'], color='blue', label='Dados Reais')
        plt.scatter(df_resultados['Paciente'], df_resultados['Previsão do Modelo'],
                    color='red', marker='x', label='Previsão do Modelo')
        plt.xlabel('Paciente')
        plt.ylabel('Classificação')
        plt.title('Comparação entre Dados Reais e Previsões do Modelo')
        plt.legend()
        plt.show()
    if scatter == False and matrix == False:
        print('Não há gráficos a serem plotados. \nSe quiser um gráfico plotado coloque o seu parâmetro como True.')
