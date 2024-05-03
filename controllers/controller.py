from models import model
from views import view


class Controller:
    def __init__(self):
        self.model = model.Model()
        self.view = view.View(self.model.modelo)

    def preprocess_and_training(self):
        self.model.preprocess()
        self.model.treinar_modelo()

    def calcular_e_salvar(self):
        self.model.tratar_dataset()
        self.model.calcular_probabilidade()
        self.model.salvar_dataset()

    def view_model(self):
        if self.view:
            self.view.view_model(
                self.model.modelo, self.model.X_teste, self.model.y_teste)


if __name__ == '__main__':
    my_controller = Controller()
    my_controller.preprocess_and_training()
    my_controller.calcular_e_salvar()
