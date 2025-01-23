from torch import nn

# Creación y definición inicial de la red neuronal
class perceptron_multicapa(nn.Module):
    def __init__(self):
        super(perceptron_multicapa, self).__init__()

        # 10 entradas, 20 (neuronas) outputs
        self.capa_oculta1 = nn.Linear(in_features=10, out_features=20)
        self.activacion_relu = nn.ReLU()

        self.capa_oculta2 = nn.Linear(in_features=20, out_features=30)
        self.activacion_tanh = nn.Tahn()

        self.capa_salida = nn.Linear(in_features=30, out_features=1)
        self.activacion_sigmoidea = nn.Sigmoid()

    def forward(self, x):
        x = self.capa_oculta1(x)
        x = self.activacion_relu(x)
        x = self.capa_oculta2(x)
        x = self.activacion_tanh(x)
        x = self.capa_salida(x)
        y = self.activacion_sigmoidea(x)
        
        return y
    
        
