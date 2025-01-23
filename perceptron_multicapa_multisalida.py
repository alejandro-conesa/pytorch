import torch

# Creación y definición inicial de la red neuronal
class perceptron_multicapa(torch.nn.Module):
    def __init__(self):
        super(perceptron_multicapa, self).__init__()

        # 10 entradas, 20 (neuronas) outputs
        self.capa_oculta1 = torch.nn.Linear(in_features=10, out_features=20)
        self.activacion_relu = torch.nn.ReLU()

        self.capa_oculta2 = torch.nn.Linear(in_features=20, out_features=30)
        self.activacion_tanh = torch.nn.Tanh()

        self.salida_1neurona = torch.nn.Linear(in_features=30, out_features=1)
        self.salida_2neuronas = torch.nn.Linear(in_features=30, out_features=2)
        self.salida_multi = torch.nn.Linear(in_features=30, out_features=5)

        self.activacion_sigmoide = torch.nn.Sigmoid()
        self.activacion_softmax = torch.nn.Softmax()

    def forward_bin_1neurona(self, x):
        x = self.capa_oculta1(x)
        x = self.activacion_relu(x)
        x = self.capa_oculta2(x)
        x = self.activacion_tanh(x)
        x = self.salida_1neurona(x)
        y = self.activacion_sigmoide(x)
        return y

    def forward_bin_2neuronas(self, x):
        x = self.capa_oculta1(x)
        x = self.activacion_relu(x)
        x = self.capa_oculta2(x)
        x = self.activacion_tanh(x)
        x = self.salida_2neuronas(x)
        y = self.activacion_softmax(x)
        return y
    
    def forward_multiclase(self, x):
        x = self.capa_oculta1(x)
        x = self.activacion_relu(x)
        x = self.capa_oculta2(x)
        x = self.activacion_tanh(x)
        x = self.salida_multi(x)
        y = self.activacion_softmax(x)
        return y
    
    def forward_multietiqueta(self, x):
        x = self.capa_oculta1(x)
        x = self.activacion_relu(x)
        x = self.capa_oculta2(x)
        x = self.activacion_tanh(x)
        x = self.salida_multi(x)
        y = self.activacion_sigmoide(x)
        return y

# Ejecución de ejemplo
tensor = torch.randn(size=(1, 10))

salida_1neurona = perceptron_multicapa().forward_bin_1neurona(tensor)
salida_2neuronas = perceptron_multicapa().forward_bin_2neuronas(tensor)
salida_multiclase = perceptron_multicapa().forward_multiclase(tensor)
salida_multietiqueta = perceptron_multicapa().forward_multietiqueta(tensor)

print(f'Binaria una neurona - Clase {"0" if salida_1neurona > 0.5 else "1"}')
print(f'Binaria dos neurona - Clase {torch.argmax(salida_2neuronas)}')
print(f'Multiclase - Clase {torch.argmax(salida_multiclase)}')
print(f'Multietiqueta - Clase {torch.where(salida_multietiqueta > 0.5)[1].tolist()}')


