import pandas as pd
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.metrics.classification import accuracy_score
from sklearn.metrics.classification import confusion_matrix
from sklearn.metrics.classification import precision_score 

X_train = pd.read_csv("pokemon_data - X (1-4).csv")
Y_train = pd.read_csv("pokemon_data - Y (1-4).csv")
X_test = pd.read_csv("pokemon_data - X (5-7).csv")
Y_test = pd.read_csv("pokemon_data - Y (5-7).csv")

Y1 = Y_train.values.astype(float)
X1 = X_train.values.astype(float)
print(X1)

Y2 = Y_test.values.astype(float)
X2 = X_test.values.astype(float)
print(X2)

# Entrenamiento
isLegendary = KNeighborsClassifier(n_neighbors = 5, metric = "minkowski", p = 1)
isLegendary.fit(X1, Y1)

# Prueba
predicción = isLegendary.predict(X2)
print("Accuracy:", accuracy_score(Y2, predicción))
print("Precision:", precision_score(Y2, predicción))
print("Confusion Matrix:")
print(confusion_matrix(Y2, predicción))

predictTest = isLegendary.predict([[131,0,680,45,95,1250000,5.8,126,131,98,99,203.0]]) # Yveltal
if(predictTest[0] == 1):
    print("The Pokémon is Legenday.")
else:
    print("The Pokémon is not Legendary.")