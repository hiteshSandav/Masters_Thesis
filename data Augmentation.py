import pandas as pd
from Code_Files.Functions import Ctgan_augment, num_data_augment, label_encode, data_split

# Read dataset 1 file into dataframe
path = "D:\\Lectures_Notes_Assignments_Tutorials\\Semester 3\\Dataset1.csv"

datafile = pd.read_csv(path)
# removes 'ID' column from the dataframe
datafile = datafile.drop(['ID'], axis = True)

# label encodes the species column
label_encode(datafile, 'Species')
# splits the data into train and test set
train_datafile, test_datafile = data_split(datafile)

# Augment data using CT-GAN in single iteration
final_df = Ctgan_augment(train_datafile, 216, 1)
final_df.to_csv(r'D:\Lectures_Notes_Assignments_Tutorials\Semester 3\datafile_1_CT.csv', index=True, header=True)

# Augment data using CT-GAN in batches
final_df = Ctgan_augment(train_datafile, 27, 8)
final_df.to_csv(r'D:\Lectures_Notes_Assignments_Tutorials\Semester 3\datafile_1_CT.csv', index=True, header=True)

# Augment data using Under-sampling-SMOTE
finaldf = num_data_augment(train_datafile, 6)
finaldf.to_csv(r'D:\Lectures_Notes_Assignments_Tutorials\Semester 3\FinalDf_USM1.csv', index = False, header = True)
