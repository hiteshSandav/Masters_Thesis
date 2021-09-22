import pandas as pd
from Code_Files.Functions import pca, biplot, screeplot, data_split, label_encode, models, surface_plot, pls

# Read dataset 1 file into dataframe
path = "D:\\Lectures_Notes_Assignments_Tutorials\\Semester 3\\Dataset1.csv"

datafile = pd.read_csv(path)
#removes 'ID' column from the dataframe
datafile = datafile.drop(['ID'], axis = True)

# label encodes the species column
label_encode(datafile, 'Species')
# splits the data into train and test set
train_datafile, test_datafile = data_split(datafile)
# label encodes the treatment column of train and test set.
label_encode(train_datafile, 'Treatment')
label_encode(test_datafile, 'Treatment')

# PCA is applied to generate components explaining 90% variance
n_components = 0.9
pca, pcacomp, testdata = pca(train_datafile, n_components, test_datafile)

# biplot and screeplot are plotted for components generated
biplot(pcacomp)
screeplot(pca)
print("Variance explained by components: \n", pca.explained_variance_ratio_)

# Surface plot is generated for top 2 components
surface_plot('svclin', pcacomp)
surface_plot('svcrbf', pcacomp)
surface_plot('svcpoly', pcacomp)
surface_plot('rf', pcacomp)

# model training and evaluation for machine learning models
models('svclin', pcacomp, testdata)
models('svcrbf', pcacomp, testdata)
models('svcpoly', pcacomp, testdata)
models('rf', pcacomp, testdata)
