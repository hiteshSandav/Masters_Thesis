import pandas as pd
from Code_Files.Functions import pca, biplot, screeplot, data_split, label_encode, models, surface_plot, plot_spectrometry, hist_density_plot

# Read dataset 1 file into dataframe
path = "D:\\Lectures_Notes_Assignments_Tutorials\\Semester 3\\Dataset1.csv"

datafile = pd.read_csv(path)
#removes 'ID' column from the dataframe
datafile = datafile.drop(['ID'], axis = True)

# label encodes the species column
label_encode(datafile, 'Species')
# splits the data into train and test set
train_datafile, test_datafile = data_split(datafile)

# Reading augmented datafile generated using CT-GAN in batches
path1 = "D:\\Lectures_Notes_Assignments_Tutorials\\Semester 3\\datafile_1_CT.csv"
finaldf = pd.read_csv(path1)
# label encodes the Treatment column
label_encode(finaldf, 'Treatment')

# Reading augmented datafile generated using CT-GAN
path2 = "D:\\Lectures_Notes_Assignments_Tutorials\\Semester 3\\datafile_1_CT_full.csv"
finaldf1 = pd.read_csv(path2)
# label encodes the Treatment column
label_encode(finaldf1, 'Treatment')

# Mass Spectrometry Plots for Original vs Augmented data
plot_spectrometry(train_datafile, finaldf)
plot_spectrometry(train_datafile, finaldf1)

# Density plots for Original vs Augmented data
hist_density_plot(train_datafile, finaldf)
hist_density_plot(train_datafile, finaldf1)

# Descriptive Statistics for original data and augmented data
print("Descriptive Statistics for training set: \n", train_datafile.describe())
print("Descriptive Statistics for Batch Augmented set: \n", finaldf.describe())
print("Descriptive Statistics for Full Augmented set: \n", finaldf1.describe())

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
