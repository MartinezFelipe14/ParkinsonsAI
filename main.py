from controller import Controller

if __name__ == '__main__':
    my_controller = Controller(view=True)
    my_controller.preprocess_and_training()
    my_controller.calcular_e_salvar()
    my_controller.view_model()
