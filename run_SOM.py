import matplotlib.pyplot as plt
import Mlp.neuron as n
import numpy as np
import pandas as pd
from Mlp.import_data import load
from sklearn.cross_validation import train_test_split

from Mlp import mlp_sparse as mpl
from Som.som import SOM


# Parameters
Ni = 0.2
n.Percepton.Hmax = 2
n.Percepton.Hmin = -2
epoch =50
hidden_layers = [8,5]
normalized = 1
alpha=100
somSize=4

# Inicialization
EDGES = dict()
SS1 = list()
R1 = list()
SS2 = list()
R2 = list()
image=np.array
# Load data
#data = load('IrisDataTrain.xls', 'Arkusz1', normalized)
data = load('../WineData.xls', 'Arkusz1', normalized)
datax = np.array(data['i'])
datay = np.array(data['o'])

X = datax
y = datay
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
alpha=len(X_train)
trainx = pd.DataFrame(X_train)
trainy = pd.DataFrame(y_train)

som = SOM(somSize, data, alpha)
for j in range(alpha):  # train
    dx = trainx.iloc[j, :]
    dy = trainy.iloc[j, :]
    som.teach(dx,dy)

image=som.rgb_image()
plt.imshow(image)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
trainx = pd.DataFrame(X_train)
trainy = pd.DataFrame(y_train)
testx = pd.DataFrame(X_test)
testy = pd.DataFrame(y_test)



network = mpl.Mpl(hidden_layers, data, EDGES,0,somSize**2)
for i in range(epoch):
    Reco1 = 0
    S1 = 0
    Reco2 = 0
    S2 = 0
    for j in range(len(trainx)):  # train
        dx = trainx.iloc[j, :]
        dx_som = som.output(dx)
        dy = trainy.iloc[j, :]
        Reco1, S1 = network.learning(dx_som, dy, j, Reco1, S1)
    for j in range(len(testx)):  # test
        dx = testx.iloc[j, :]
        dx_som = som.output(dx)
        dy = testy.iloc[j, :]
        Reco2, S2 = network.calc_output(dx_som, dy, j, Reco2, S2)
    n.Percepton.Ni *= 0.99
    print(Reco2 / len(testy))
    print(Reco1 / len(trainx))

    SS2.append(S2)
    R2.append(Reco2 / len(testy))

    SS1.append(S1)
    R1.append(Reco1 / len(trainy))

# Display
plt.subplot(3, 1, 1)
plt.plot(R1)
plt.title("Percent of guesses train")
plt.ylim(0, 1)
plt.subplot(3, 1, 2)
plt.plot(SS1)
plt.title("Sum of squares1")
plt.subplot(3, 1, 3)
plt.plot(R2)
plt.title("Percent of guesses test")
plt.show()