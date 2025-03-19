'''
!_TeamSiance/Santé/doc/Illustration.jpeg [« Crédit image : Google AI »]

*Maîtriser le diabète : comment l'IA rend plus efficaces les actions conventionnelles ?
**Motivations :
Pour trouver une meilleure réponse au diabète, j’ai entrepris une campagne de mesures portant sur la composition de mes aliments en glucides, lipides, protéines, fibres, énergie, indice glycémique et charge glycémique. À chaque repas, je relève ces mesures, ainsi que ma glycémie prépandriale et postprandiale. Pendant 2 semaines, j’ai constitué un dataset. Que j’ai analysé par les techniques statistiques et d’analyse de données, afin de normaliser ces données et mesurer leur cohérence et la corrélation entre les différentes variables.
J’ai ensuite appliqué les techniques d’apprentissage afin d’élaborer le modèle le plus pertinent pour prédire et expliquer la glycémie postprandiale 2 heures après chaque repas à partir des autres variables indépendantes. 
#1. Préparation de l'environnement et des données
* Bibliothèques nécessaires :
   * pandas pour la manipulation des données.
   * scikit-learn pour les algorithmes d'apprentissage automatique.
   * numpy pour les opérations numériques.
   * matplotlib et seaborn pour la visualisation (optionnel).'
'''
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

sante_directory = r"C:\\Users\\admin\\anaconda3\\envs\\ATMAN\\_TeamSiance\\Santé"
#Forcer le répertoire courant
os.chdir(sante_directory)

'''
# Chargement du dataset et pré-traitement des données
* Assurez-vous que votre jeu de données est dans un format compatible (CSV, Excel, etc.).
* Utilisez pandas pour charger les données dans un DataFrame.
Comme c'est toujours le cas en IA, le temps consacré à l'exploration et nettoyage des données, la mise en forme, mise en cohérence et normalisation des données
est la tâche qui demande beaucoup d'efforts et d'analyse pour aboutir à des modèles d'apprentissage
performants.
* Vérifiez les types de données, les valeurs manquantes, les valeurs aberrantes, etc.
'''

data = pd.read_csv("doc\\Suivi_GL.csv", sep=None, engine="python", decimal=",")

# Supprimer les espaces ou caractères étranges dans chaque cellule
for col in data.columns:
    if data[col].dtype == "object":
        data[col] = data[col].str.strip()

# Remplacer les virgules par des points dans les colonnes numériques
cols_to_convert = ['Kcal', 'PRO', 'GLU', 'LIP', 'IG', 'CG', 'GPO']
for col in cols_to_convert:
    if data[col].dtype == "object":
        data[col] = data[col].str.replace(',', '.').astype(float)

# Set the backend before importing pyplot
#matplotlib.use("TkAgg")

'''
*Investigation et analyse statistique des données
Utilisation les bibliothèques matplotlib et seaborn.
Cette étape est très importante pour identifier les variables indépendantes (features) les plus influentes sur 
les variables qu'on cherche à prédire. N'hésitez pas à afficher les statistiques descriptives print(data.describe()).
Visualisation des données
if not data.empty and len(data.columns) > 0:
    plt.figure(figsize=(10, 6))
    # Define vars to use all columns for the pairplot
    vars = data.columns  # Use all columns
    print(f'vars : {vars}')
    # Create pairplot
    sns.pairplot(data, vars=vars)
        
    # Alternatively, use PairGrid for more customization
    g = sns.PairGrid(data, vars=vars)
    g.map_upper(sns.scatterplot)
    g.map_lower(sns.kdeplot, cmap="Blues_d")
    g.map_diag(sns.histplot, kde_kws={"color": "k"})
        
    # Ensure the plot is displayed
    plt.show()
# Vérifier les corrélations
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.show()
'''
''' 
#2. Sélection des caractéristiques et des cibles
# Séparer les features (X) et la cible (y)
* Caractéristiques (X) : ['Kcal','PRO','GLU', 'LIP', 'IG','CG','GPR']
* Cibles (y) : GPO
'''
X = data[['Kcal','PRO','GLU', 'LIP', 'IG','CG','GPR']]
y = data['GPO']

# Normaliser les caractéristiques
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

'''
#4. Choix et entraînement du modèle
**La régression linéaire est un bon point de départ, mais d'autres modèles peuvent être envisagés (forêts aléatoires, réseaux de neurones, etc.).
model_gl = LinearRegression()
model_gl.fit(X_train, y_train)
La visualisation des données effectuées dans les phases précédentes ne montrait pas de corrélation forte entre les données.
Ce modèle est donc de prime abord voué à l'échec.
Résultats du modèle model_gl : GPO : MSE = 757.7379903131332, R2 = -1.790075447608959
Ces métriques indiquent que ton modèle de prédiction de glycémie (GPO) est de mauvaise qualité. Voici pourquoi :

MSE (Mean Squared Error) = 757.74 signifie que l'erreur quadratique moyenne est élevée.
Une valeur élevée du MSE indique que les prédictions du modèle sont loin des valeurs réelles.

R2 (Coefficient de détermination) = −1.79 est un très mauvais score. Un bon modèle a un un R2 proche de 1 (1 signifie une prédiction parfaite).
Un R2 négatif signifie que le modèle est pire qu'une simple moyenne des valeurs observées. En d’autres termes, prendre la moyenne des glycémies comme prédiction donnerait un meilleur résultat que ton modèle actuel.

**Le modèle Random Forest
model_gl_rf = RandomForestRegressor(n_estimators=100, random_state=42)
Pourquoi Random Forest ?
- Il gère bien les relations complexes entre variables.
- Il est robuste aux outliers et ne nécessite pas trop de preprocessing.
- Il permet d’évaluer l’importance des variables.

Résultats du modèle model_gl_rfr : GPO : MSE = 172.79431666666656, R2 = 0.36375213255599914

# Initialiser et entraîner le modèle XGBoost
model_gl_xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model_gl_xgb.fit(X_train, y_train)

#Explication des paramètres de XGBoost
n_estimators=100 : Nombre d'arbres (plus il y en a, plus le modèle est précis mais lent).
learning_rate=0.1 : Taux d'apprentissage, contrôle la vitesse d'ajustement du modèle.
max_depth=5 : Profondeur des arbres, empêche l'overfitting.
Résultats : GPO : MSE = 843.3014899948126, R2 = -2.105129757575254

# Initialiser et entraîner le modèle LightGBM
model_lgb = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model_lgb.fit(X_train, y_train)

from sklearn.utils.validation import check_is_fitted
try:
    check_is_fitted(model_lgb)
    print("Modèle bien entraîné, prêt pour la prédiction.")
except:
    print("Erreur : Le modèle n'est pas encore entraîné.")

# Évaluation du modèle LightGBM
y_pred_lgb = model_lgb.predict(X_test)
mse_lgb = mean_squared_error(y_test, y_pred_lgb)
r2_lgb = r2_score(y_test, y_pred_lgb)
print(f'LightGBM : MSE = {mse_lgb}, R2 = {r2_lgb}')

# Sauvegarder le modèle LightGBM
joblib.dump(model_lgb, "model_lgb.pkl")
'''

#créer les ensembles d'entrainement et de test 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vérification avant entraînement
if X_train.isna().sum().sum() > 0 or y_train.isna().sum() > 0:
    raise ValueError("Il y a des valeurs manquantes dans X_train ou y_train")

# Convertir les données en tableaux NumPy
X_train_np = np.array(X_train)
X_test_np = np.array(X_test)
y_train_np = np.array(y_train).reshape(-1, 1)
y_test_np = np.array(y_test).reshape(-1, 1)

# Convertir les données en tenseurs PyTorch
X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Définir le modèle de réseau de neurones
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model_nn = NeuralNetwork()

# Définir la fonction de perte et l'optimiseur
criterion = nn.MSELoss()
optimizer = optim.Adam(model_nn.parameters(), lr=0.001)

# Entraîner le modèle
num_epochs = 700
for epoch in range(num_epochs):
    model_nn.train()
    optimizer.zero_grad()
    outputs = model_nn(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

'''#5. Évaluation du modèle
 Utilisez des métriques appropriées (erreur quadratique moyenne, coefficient de détermination, etc.).
'''
model_nn.eval()
with torch.no_grad():
    y_pred_nn = model_nn(X_test_tensor)
    mse_nn = mean_squared_error(y_test_tensor.numpy(), y_pred_nn.numpy())
    r2_nn = r2_score(y_test_tensor.numpy(), y_pred_nn.numpy())
    print(f'Neural Network (PyTorch) : MSE = {mse_nn}, R2 = {r2_nn}')

'''
# Importance des features, applicable seulement pour certains modèles, mais pas les RNN
feature_importances = pd.DataFrame({"Feature": X.columns, "Importance": model_nn.feature_importances_})
print(feature_importances.sort_values(by="Importance", ascending=False))
'''
# Sauvegarder le modèle de réseau de neurones
os.makedirs("models", exist_ok=True)
torch.save(model_nn.state_dict(), "models/model_nn.pth")

'''
6. Amélioration du modèle (optionnel)
 * Ajustez les hyperparamètres, essayez d'autres modèles, ajoutez des caractéristiques, etc.
Conseils supplémentaires
 * Visualisez les résultats pour mieux comprendre les performances du modèle.
 * Considérez la normalisation ou la standardisation des caractéristiques.
 * Pour des modèles plus complexe, l'utilisation de librairies comme Tensorflow ou Pytorch pourrait être bénéfique.
'''

import streamlit as st
import numpy as np
import pickle  # Ou joblib selon le format de sauvegarde de votre modèle

# Charger le modèle entraîné (assurez-vous d'avoir le fichier du modèle dans le même dossier)
model_nn.load_state_dict(torch.load("models/model_nn.pth"))
model_nn.eval()

'''
with open('model_lgb.pkl', 'rb') as file:
   model = joblib.load("model_lgb.pkl")'
'''

# Titre de l'application
st.title("Application de Prédiction avec Streamlit")

# Entrée utilisateur
st.header("Saisissez les paramètres")

# Ajouter des champs d'entrée pour chaque paramètre (en fonction de votre modèle)GLU, PRO, LIP, KCA.

GLU = st.number_input("Glucides", value=0.0)
PRO = st.number_input("Protéines", value=0.0)
LIP = st.number_input("Lipides", value=0.0)
Kcal = st.number_input("Kcal", value=0.0)
IG = st.number_input("Indice Glycémique",value=0.0)
CG = st.number_input("Charge Glycémique",value=0.0)
GPR = st.number_input("Glycémie Prépandriale",value=0.0)

# Bouton pour faire une prédiction
if st.button("Prédire"):
    # Mettre les paramètres dans le bon format pour le modèle
    input_data = np.array([[Kcal,PRO,GLU,LIP,IG,CG,GPR]]) # Adapter selon le nombre de paramètres
    input_data_scaled = scaler.transform(input_data)
    input_data_tensor = torch.tensor(input_data_scaled, dtype=torch.float32)#prediction = model.predict(input_data)  # Effectuer la prédiction
    prediction_tensor = model_nn(input_data_tensor)
    prediction = float(prediction_tensor.item())  # On convertit en float natif
    # Afficher le résultat
    #st.success(f"Le résultat de la prédiction est : {prediction[0]}")
      # Définir la couleur en fonction des seuils
    if prediction < 1.20:
        color = "green"
        message = f"Le résultat de la prédiction est : {prediction:.2f} (Vert : Acceptable)"
    elif 1.21 <= prediction <= 1.35:
        color = "orange"
        message = f"Le résultat de la prédiction est : {prediction:.2f} (Orange : Attention)"
    else:
        color = "red"
        message = f"Le résultat de la prédiction est : {prediction:.2f} (Rouge : Critique)"

    # Afficher le message coloré
    st.markdown(f"<h3 style='color: {color};'>{message}</h3>", unsafe_allow_html=True)
