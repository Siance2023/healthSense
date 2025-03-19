'''
Visualize the data in a pairplot using seaborn and matplotlib libraries.
'''
import pandas as pd
import seaborn as sns
import matplotlib

# Set the backend before importing pyplot
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# Load the data
file_path = r'c:\\Users\\admin\\anaconda3\\envs\\ATMAN\\_TeamSiance\\Santé\\Suivi_GL.csv'
data = pd.read_csv(file_path, delimiter=';', decimal=',')


# Supprimer les espaces ou caractères étranges dans chaque cellule
for col in data.columns:
    if data[col].dtype == "object":
        data[col] = data[col].str.strip()

# Remplacer les virgules par des points dans les colonnes numériques
cols_to_convert = ['Kcal', 'PRO', 'GLU', 'LIP', 'IG', 'CG', 'GPR','GPO']
for col in cols_to_convert:
    if data[col].dtype == "object":
        data[col] = data[col].str.replace(',', '.').astype(float)

#Visualisation des résultats
import matplotlib
matplotlib.use("TkAgg") # Change le backend pour l'affichage

'''# Distribution de chaque colonne
sns.pairplot(data)
plt.show()
'''
# Heatmap pour les corrélations
plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Matrice de corrélation")
plt.show()

'''
# Exemple : Scatter plot entre deux variables
sns.scatterplot(data=data, x="GLU", y="GL")
plt.title("GLU vs GL")
plt.show()

# Exemple : Boxplot pour observer la distribution d'une variable
sns.boxplot(data=data, y="GL")
plt.title("Distribution de GL")
plt.show()
'''

if __name__ == "__main__":
    # Display the first few rows of the dataframe
    print(data.head())

    # Print the columns of the dataframe
    print("Columns in the DataFrame:", data.columns)

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
    else:
        print("The DataFrame is empty or has no columns to plot.")