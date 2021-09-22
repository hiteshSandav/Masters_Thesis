import pandas as pd
from Code_Files.Functions import data_split, label_encode, pls, scores_plot

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

# Reading augmented datafile generated using CT-GAN
path2 = "D:\\Lectures_Notes_Assignments_Tutorials\\Semester 3\\datafile_1_CT_full.csv"
finaldf1 = pd.read_csv(path2)

# PLS-DA is applied for generating 2 components
n_components = 2
scores_traindf, scores_testdf = pls(finaldf, test_datafile, n_components)

# scores plot is plotted for 2 components generated
scores_plot(scores_traindf)

