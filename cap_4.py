import torch
import torch.optim.adam
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Al ser el primer caso de estudio está prácticamente copiado del apéndice A

class perceptron_multicapa(torch.nn.Module):
    def __init__(self):
        super(perceptron_multicapa, self).__init__()

        self.backbone = torch.nn.Sequential(
            torch.nn.Linear(in_features=784, out_features=512),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=512, out_features=10)
        )

    def forward(self, input):
        input = torch.flatten(input.squeeze(1), start_dim=1, end_dim=2) 
        # squeeze quita la dimensión 1 (la segunda, la primera es 0)
        # el método flatten combina las dimensiones desde el inicio hasta el final (1 y 2, es decir, de 28x28 a 784)
        return self.backbone(input)

def epoca_train(modelo, train_loader, optimizador, funcion_perdida, dispositivo):
    epoca_perdidas = []
    for i, (imagenes, etiquetas) in enumerate(train_loader):
        imagenes = imagenes.to(dispositivo)
        etiquetas = etiquetas.to(dispositivo)

        optimizador.zero_grad()

        salidas = modelo.forward(imagenes)
        perdida = funcion_perdida(salidas, etiquetas)

        perdida.backward()
        optimizador.step()

        epoca_perdidas.append(perdida.item())
    
    return sum(epoca_perdidas)/len(epoca_perdidas)

def epoca_test(modelo, test_loader, dispositivo):
    modelo.eval()

    with torch.no_grad():
        correcto = 0
        total = 0

        for imagenes, etiquetas in test_loader:
            imagenes = imagenes.to(dispositivo)
            etiquetas = etiquetas.to(dispositivo)

            salidas = modelo(imagenes)
            salidas = salidas.softmax(dim=1) # comprobar si esta parte es correcta (nombre variables)

            prediccion = salidas.argmax(dim=1)

            total += etiquetas.size(0)
            correcto += (prediccion == etiquetas).sum().item() # comprobar que hace esta línea
        
        exactitud = 100 * correcto / total

    modelo.train() # devuelve al modelo al modo de entrenamiento
    return exactitud

# Carga y transformación de los sets en iterables
# La normalización de los datos y conversión a tensor se realiza con ToTensor()
# La conversión a one-hot no es necesaria, se encarga la función de pérdida
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
train_set, val_set = torch.utils.data.random_split(mnist_trainset, [50000, 10000])
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=32, shuffle=True)

mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(mnist_testset, batch_size=32, shuffle=False)

# Creación del modelo
modelo = perceptron_multicapa()
print(modelo)

dispositivo = torch.device('mps' if torch.mps.is_available() else 'cpu')
modelo.to(dispositivo)
print(f'Usando: {dispositivo}')

entropia_categorica_cruzada = torch.nn.CrossEntropyLoss()

optimizador = torch.optim.Adam(modelo.parameters())

# Entrenamiento de la red
train_accs = []
val_accs = []
epocas = 15

for epoca in range(epocas):
    train_perdida = epoca_train(modelo, train_loader, optimizador, entropia_categorica_cruzada, dispositivo)
    train_acc = epoca_test(modelo, train_loader, dispositivo)
    train_accs.append(train_acc)

    val_acc = epoca_test(modelo, val_loader, dispositivo)
    val_accs.append(val_acc)

    print(f'Época {epoca+1}/{epocas},\
            pérdida={train_perdida:.4f},\
            exactitud_train={train_acc:.2f}%,\
            exactitud_validacion={val_acc:.2f}%')

print('Fin del entrenamiento')

# Mostrar gráfico
plt.plot(train_accs)
plt.plot(val_accs)
plt.xlabel('Época')
plt.ylabel('Exactitud (%)')
plt.legend(['Entrenamiento', 'Validación'])
plt.show()

# Exactitud en los datos de test
test_acc = epoca_test(modelo, test_loader, dispositivo)
print(f'Exactitud test: {test_acc:.2f}%')

# Guardar modelo como diccionario de python
torch.save(modelo.state_dict(), 'pesos.pth')

nuevo_modelo = perceptron_multicapa()
nuevo_modelo.load_state_dict(torch.load('pesos.pth'))
nuevo_modelo.to(dispositivo)

test_acc = epoca_test(nuevo_modelo, test_loader, dispositivo)
print(f'Exactitud test: {test_acc:.2f}%')