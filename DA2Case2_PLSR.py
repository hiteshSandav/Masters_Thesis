import pandas as pd
from Code_Files.Functions import plsr, report

# Read dataset 2 file into dataframe
path = "D:\\Lectures_Notes_Assignments_Tutorials\\Semester 3\\Dataset2.xlsx"
datafile = pd.read_excel(path, sheet_name='Sheet1', engine = 'openpyxl')
print(datafile)

# to report dataset Statistics (duplicate values, missing values etc)
report(datafile)

# Creation of Feature set and Target set
feature = datafile.drop(['Fat', 'Moisture'], axis = 1)
target = datafile[['Fat', 'Moisture']]

# Applying PLSR to select optimum number of components using predict function
# To predict Fat content
plsr(feature, target["Fat"], 40, plot_components=True)
# To predict Moisture content
plsr(feature, target["Moisture"], 40, plot_components=True)
