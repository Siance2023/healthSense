# Maîtriser le diabète : Comment l'IA peut contribuer à l'amélioration du contrôle glycémique ?
![Illustration](doc/Illustration.jpeg)  
---Crédit Image : Google AI---

## 🔎 Motivations
Afin d'analyser l'impact de la composition et de l'équilibre des repas sur la glycémie, j'envisage de m'appuyer sur les avancées de l'Intelligence Artificielle.
L'objectif est d'entrainer un modèle d'apprentissage par réseaux de neurones pour tenir compte de la nature des données, peu corrélées les unes et autres, et surtout de la complexité de la tâche. Tous les experts du domaine consultés en conviennent. Il s'agit ici de construire un modèle capable de prédire la glycémie postprandiale à partir des valeurs nutritionnelles des repas.

## 1. Premier défi : collecter des données précises et en quantité suffisante
Pour cela, il fallait d'abord collecter des mesures, beaucoup de mesures ! J'ai donc décidé de conduire une campagne systématique de mesures portant sur les valeurs nutritionnelles de tous mes repas (4 repas par jour : petit-déjeuner, déjeuner, snack et diner), ainsi que sur les valeurs des glycémies pré- et postprandiales (soit 8 prélèvements par jour). La tâche la plus ardue consiste à déterminer les quantités des aliments constitutifs des repas et de retrouver leurs apports nutritionnels (glucides, protéines, lipides, fibres, énergie (kcal), index glycémique et charge glycémique). Bien que ces informations soient documentées, les sources sont souvent incomplètes et divergent notablement. Le premier challenge consiste donc à collecter des données fiables et les plus précises possibles.
Cette compagne a été conduite sur une période de 2 semaines.

## 📝 Données collectées :
- Nutriments : Glucides, Lipides, Protéines, Fibres
- Indice glycémique & charge glycémique
- Energie : kcal
- Glycémie pré- et postprandiale

✅ Premières constatations significatives sur la trajectoire des glycémies
Très vite, on intégre et on mémorise les quantités d'aliments constitutifs des repas et les apports des produits en termes nutritionnels. Cette connaissance conduit à composer des repas équilibrés. Cet équilibre est une variable très influente sur la limitation des pics glycémiques. Mes glycémies postprandiales ont chuté de l'ordre de 30% au bout d'une semaine. En fait, tout se passe comme si pour prendre des actions efficaces, il suffit de prendre des mesures. En somme, **Pour prendre des mesures, prenez des mesures **!

## 🛠 Préparation des données
```python
Importer les bibliothèques nécessaires
import os, sys
import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import torch
import torch.nn as nn
import torch.optim as optim
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
```
### 🛠 Chargement et nettoyage des données
```python
# Charger les données
data = pd.read_csv("Fichier.csv", sep=None, engine="python", decimal=",") #Votre Fichier.csv

# Nettoyage
for col in data.columns:
    if data[col].dtype == "object":
        data[col] = data[col].str.strip().str.replace(',', '.').astype(float)
```

### 📊 Visualisation des corrélations entre features
```python
import seaborn as sns
import matplotlib.pyplot as plt

corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.show()
```

![Illustration](logs/Figure_1.png) 
---Corrélation entre les features---

Nous constatons une corrélation faible entre les données.

### 🎯 Choix des variables et normalisation

```python
X = data[['Kcal', 'PRO', 'GLU', 'LIP', 'IG', 'CG', 'GPR']]
y = data['GPO']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

## 2. 🎉 Essais d'entraînement de quelques modèles
📊 Régression Linéaire (Baseline)
```python
model_gl = LinearRegression()
model_gl.fit(X_scaled, y)
y_pred = model_gl.predict(X_scaled)

print("MSE:", mean_squared_error(y, y_pred))
print("R2:", r2_score(y, y_pred))
```

⚠️ Problème : Score R² = -1.7 est un très mauvais score !
Un bon modèle doit avoir un R² proche de 1 (1 signifie une prédiction parfaite). Un R² négatif signifie que le modèle est pire qu'une simple moyenne des valeurs observées. Au vu de la faible corrélation entre les données, ce résultat n'est pas surprenant.

💡XGBoost
```python
model_xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model_xgb.fit(X_scaled, y)
y_pred_xgb = model_xgb.predict(X_scaled)
```
❌ **Résultat décevant** (R² négatif), donc pas adapté non plus pour notre problème.

🌟 Random Forest
```python
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_scaled, y)
y_pred_rf = model_rf.predict(X_scaled)
```
⚠️Amélioration : R² passe à 0.36
Les modèles statistiques se sont avérés peu indiqués pour traiter ce problème, nous nous sommes donc tournés vers les réseaux de neurones.

## 3.🔍 Entrainement d'un réseau de neurones avec PyTorch
```python
import torch
import torch.nn as nn
import torch.optim as optim

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(X_scaled.shape[1], 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Initialisation
model_nn = NeuralNetwork()
criterion = nn.MSELoss()
optimizer = optim.Adam(model_nn.parameters(), lr=0.001)

# Entraînement
num_epochs = 500
for epoch in range(num_epochs):
    model_nn.train()
    optimizer.zero_grad()
    outputs = model_nn(torch.tensor(X_scaled, dtype=torch.float32))
    loss = criterion(outputs, torch.tensor(y.values, dtype=torch.float32).view(-1, 1))
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss. Item():.4f}')
```
✅ Le réseau de neurones a donné les meilleurs résultats !
Comme hyperparamètres, ceux qui conduisent au meilleur profil de la loss sont dans notre cas : num_epochs = 500 et learning-rate = 0,001. Le profil de la loss suggère que le modèle apprend bien.

🔍Les premiers résultats obtenus montrent une prévalence des protéines sur les autres facteurs sur la glycémie postprandiale (GPO). Ils semblent suggérer aussi que le contrôle de la glycémie repose sur un équilibre entre protéines et glucides. L'objectif des prochaines étapes de ce projet est de confirmer ce résultat et de le quantifier.

## 4. 🛠 Création de l'application Streamlit
Une application streamlit a été développée afin de saisir et de soumettre les paramères GLU, PRO, LIP, IG, CG, GPR afin de prédire GPO, glycémie postprandiale.
```python
st.title("Prédiction de la Glycémie Postprandiale")
GLU = st.number_input("Glucides")
PRO = st.number_input("Protéines")
LIP = st.number_input("Lipides")
IG = st.number_input("Indice Glycémique")
CG = st.number_input("Charge Glycémique")
GPR = st.number_input("Glycémie Prépandriale")

if st.button("Prédire"):
    input_data = torch.tensor([[GLU, PRO, LIP, IG, CG, GPR]], dtype=torch.float32)
    prediction = model_nn(input_data).item()
    st.success(f"Glycémie postprandiale prédite : {prediction:.2f} mg/dL")
```

✅ Les prévisions sont conformes aux mesures de glycémie réalisées après les repas.

## 🔎 Conclusion
✅ Les réseaux de neurones sont adaptés à ce domaine. Parmi tous les modèles expérimentés, ils sont les plus performants et prometteurs.

✅ Des résultats encourageants
Bien que les modèles soient perfectibles, principalement en enrichissant le dataset, les premiers résultats sont encourageants. Les glycémies prédites sont conformes à celles qui sont effectivement mesurées.
L'IA peut apporter des avancées remarquables dans ce domaine de santé publique, en découvrant d'autres pistes pour accompagner les personnes DT1 ou DT2, notamment autour de l'équilibre des repas et pas seulement en adoptant des régimes restrictifs et très contraignants de limitation de tel ou tel type d'aliments.

✅ Nécessité d'augmenter et de diversifier les données
Comme l'expérience est menée à partir des données d'une seule personne, qui s'alimente de façon équilibrée, sans excès et à heures fixes, qui s'adonne de plus à une activité sportive régulière, donc sujet à une errance glycémique modérée, les données GPO ne varient pas énormément entre les repas. Le modèle a tendance à prédire des valeurs qui sont voisines des données d'entrainement. Il faudrait donc tester avec d'autres profils de personnes pour déterminer la capacité de généralisation du modèle. C'est l'objectif des phases ultérieures de ce projet.

✅ Prochaines étapes ?
- Améliorer l'entraînement du réseau neuronal avec plus de données ! 🚀
- Augmenter le nombre de features (horaire des repas, activité physique) et améliorer la quantité et la qualité des données.

A ce sujet, je fais un appel pour constituer une communauté motivée pour enrichir le dataset, qui serait public et anonyme.