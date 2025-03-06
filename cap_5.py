import torch
import torch.optim.adam
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = torch.nn.Conv2d(1, 6, kernel_size=5, padding='same')
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv2 = torch.nn.Conv2d(6, 16, kernel_size=5, padding='same')
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2)

        self.flatten = torch.nn.Flatten()
        self.capa1 = torch.nn.Linear(in_features=784, out_features=120)
        self.capa2 = torch.nn.Linear(in_features=120, out_features=84)
        self.salida = torch.nn.Linear(in_features=84, out_features=10)

        self.relu = torch.nn.ReLU()
    
    def forward(self, x):
        # extracción
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)

        # clasificación
        x = self.flatten(x)
        x = self.capa1(x)
        x = self.relu(x)
        x = self.capa2(x)
        x = self.relu(x)
        y = self.salida(x)
        return y
    

def epoca_train(modelo, train_loader, optimizador, funcion_perdida, dispositivo):
    epoca_perdidas = []
    for _, (imagenes, etiquetas) in enumerate(train_loader):
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
            salidas = salidas.softmax(dim=1)
            prediccion = salidas.argmax(dim=1)

            total += etiquetas.size(0)
            correcto += (prediccion==etiquetas).sum().item()
        
        exactitud = 100 * correcto / total

    modelo.train()
    return exactitud

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=64, shuffle=True)

mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(mnist_testset, batch_size=64, shuffle=False)

dispositivo = torch.device('mps' if torch.mps.is_available() else 'cpu')
modelo = CNN()
modelo.to(dispositivo)

entropia_categorica_cruzada = torch.nn.CrossEntropyLoss()
optimizador = torch.optim.Adam(modelo.parameters())

train_accs = []
epocas = 15

for epoca in range(epocas):
    train_perdida = epoca_train(modelo, train_loader, optimizador, entropia_categorica_cruzada, dispositivo)
    train_acc = epoca_test(modelo, train_loader, dispositivo)
    train_accs.append(train_acc)

    print(f'Época {epoca+1}/{epocas},\
            pérdida={train_perdida:.4f},\
            exactitud_train={train_acc:.2f}%')

plt.plot(train_accs)
plt.xlabel('Época')
plt.ylabel('Exactitud (%)')
plt.legend(['Entrenamiento'])
plt.show()

test_acc = epoca_test(modelo, test_loader, dispositivo)
print(f'Exactitud test: {test_acc:.2f}%')
torch.save(modelo.state_dict(), 'pesos_cap_5.pth')