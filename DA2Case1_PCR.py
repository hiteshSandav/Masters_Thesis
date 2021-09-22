import pandas as pd
from Code_Files.Functions import pcr, report

# Read dataset 2 file into dataframe
path = "D:\\Lectures_Notes_Assignments_Tutorials\\Semester 3\\Dataset2.xlsx"
datafile = pd.read_excel(path, sheet_name='Sheet1', engine = 'openpyxl')
print(datafile)

# to report dataset Statistics (duplicate values, missing values etc)
report(datafile)

# Creation of Feature set and Target set
feature = datafile.drop(['Fat', 'Moisture'], axis = 1)
target = datafile[['Fat', 'Moisture']]

# Applying PCR to select optimum number of components using various machine learning models
# To predict Fat content
pcr('svclin', feature, target["Fat"], pc=40)
pcr('svcrbf', feature, target["Fat"], pc=40)
pcr('svcpoly', feature, target["Fat"], pc=40)
pcr('lr', feature, target["Fat"], pc=40)
pcr('en', feature, target["Fat"], pc=40)
pcr('dtr', feature, target["Fat"], pc=40)

# To predict Moisture content
pcr('svclin', feature, target["Moisture"], pc=40)
pcr('svcrbf', feature, target["Moisture"], pc=40)
pcr('svcpoly', feature, target["Moisture"], pc=40)
pcr('lr', feature, target["Moisture"], pc=40)
pcr('en', feature, target["Moisture"], pc=40)
pcr('dtr', feature, target["Moisture"], pc=40)
