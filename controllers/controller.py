from models import model
from views import view


class Controller:
    def __init__(self):
        self.model = model.Model()

    def preprocess_and_training(self):
        self.model.preprocess()
        self.model.treinar_modelo()

    def calcular_e_salvar(self):
        self.model.tratar_dataset()
        self.model.calcular_probabilidade()
        self.model.calcular_resultado()
        self.model.salvar_dataset()

    def view_model(self, scatter=True, matrix=True):
        # plota o modelo com a função do arquivo view
        view.view_model(self.model.pipe, self.model.X_teste,
                        self.model.y_teste, scatter=scatter, matrix=matrix)
