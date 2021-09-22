import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.decomposition import PCA
from Code_Files.Functions import plsr

#function to return loading matrix after applying PCA to generate 10 components
def pcr(features):
    Xstd = StandardScaler().fit_transform(features)

    pca = PCA(n_components = 10)
    pca.fit_transform(Xstd)

    print("Variance explained by each principal component: \n", np.round(pca.explained_variance_ratio_, decimals = 3))
    loading = pca.components_.T * np.sqrt(pca.explained_variance_)

    col = []
    for i in range(1,11):
        col.append("PC"+str(i))

    loading_matrix = pd.DataFrame(loading, index=features.columns, columns=col)
    loadings = pd.DataFrame(pca.components_.T, columns=col)

    return loading_matrix, loadings


# Read dataset 2 file into dataframe
path = "D:\\Lectures_Notes_Assignments_Tutorials\\Semester 3\\Dataset2.xlsx"
datafile = pd.read_excel(path, sheet_name='Sheet1', engine = 'openpyxl')

# Creation of Feature set and Target set
feature = datafile.drop(['Fat', 'Moisture'], axis = 1)
target = datafile[['Fat', 'Moisture']]

# Applying PCR to find top loadings
loading_matrix, loadings = pcr(feature)

#processing loadings matrix to store loadings upto 2 digits after decimal place
loading_matrix['PC1'] = loading_matrix['PC1'].round(decimals=2)
loadings['PC1'] = loadings['PC1'].round(decimals=2)
print(loading_matrix['PC1'])

# Reduced datafile containing top loadings from component 1
reduced_cols = loading_matrix.index[loading_matrix['PC1'] >= 1].tolist()
X = datafile[reduced_cols]

# Applying PLSR on reduced dataset to select optimum number of components using predict function
# To predict Fat content
plsr(X, target['Fat'], 40, plot_components=True)
# To predict Moisture content
plsr(X, target['Moisture'], 40, plot_components=True)
