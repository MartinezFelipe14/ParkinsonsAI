from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def view_model(modelo_principal, X_teste, y_teste):
    # fazer uma previsão do modelo a fins de teste
    previsao_modelo_principal = modelo_principal.predict(X_teste)

    # testar a previsão do modelo
    print(previsao_modelo_principal)

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
