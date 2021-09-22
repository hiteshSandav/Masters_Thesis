import pandas as pd
from Code_Files.Functions import pca, biplot, screeplot, data_split, label_encode, models, surface_plot, plot_spectrometry, hist_density_plot, num_data_augment

# Read dataset 1 file into dataframe
path = "D:\\Lectures_Notes_Assignments_Tutorials\\Semester 3\\Dataset1.csv"

datafile = pd.read_csv(path)
#removes 'ID' column from the dataframe
datafile = datafile.drop(['ID'], axis = True)

# label encodes the species column
label_encode(datafile, 'Species')
# splits the data into train and test set
train_datafile, test_datafile = data_split(datafile)

# Reading augmented datafile generated using Under-Sampling_SMOTE in batches
path1 = "D:\\Lectures_Notes_Assignments_Tutorials\\Semester 3\\FinalDf_USM1.csv"
finaldf = pd.read_csv(path1)
# label encodes the Treatment column for augmented train and test set
label_encode(finaldf, 'Treatment')
label_encode(test_datafile, 'Treatment')

# Mass Spectrometry Plots for Original vs Augmented data
plot_spectrometry(train_datafile, finaldf)

# Density plots for Original vs Augmented data
hist_density_plot(train_datafile, finaldf)

# Descriptive Statistics for original data and augmented data
print("Descriptive Statistics for training set: \n", train_datafile.describe())
print("Descriptive Statistics for Augmented set: \n", finaldf.describe())

# PCA is applied to generate components explaining 90% variance
n_components = 0.9
pca, pcacomp, testdata = pca(finaldf, n_components, test_datafile)

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
