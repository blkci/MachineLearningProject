# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 02:59:57 2023

@author: muham
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 02:45:42 2023

@author: muham
"""

# Import the necessary modules
from osgeo import gdal
import numpy as np
import pandas as pd

# Open the GeoTIFF files using GDAL
datasetTrainingGT = gdal.Open(r'C:\Users\muham\Downloads\Project-20230123T143514Z-001\Project\S2A_MSIL1C_20220516_Train_GT.tif')

# Read the data from the first GeoTIFF file into a NumPy array
trainGT2d = datasetTrainingGT.ReadAsArray()
trainGT2d = np.transpose(trainGT2d)
# Convert the 2-dimensional NumPy arrays into 2-dimensional arrays with rows and columns
trainGT1d = trainGT2d.reshape(-1, 1)

# Convert the combined array into a Pandas DataFrame
dfTrainLabels = pd.DataFrame(trainGT1d)

# Export the DataFrame as a CSV file
# dfTrainLabels.to_csv('train.csv', index=False)
np.save('train_gt.npy', trainGT1d)

datasetTraining = gdal.Open(r'C:\Users\muham\Downloads\Project-20230123T143514Z-001\Project\S2A_MSIL1C_20220516_TrainingData.tif')

# Read the data from the first GeoTIFF file into a NumPy array
dataTraing = datasetTraining.ReadAsArray()
dataTraing = np.transpose(dataTraing)
# Convert the 2-dimensional NumPy arrays into 2-dimensional arrays with rows and columns
dataTraining1d = dataTraing.reshape(dataTraing.shape[0] * dataTraing.shape[1], -1)
dfTrain = pd.DataFrame(dataTraining1d)

final_data = np.concatenate([trainGT1d, dataTraining1d], axis=1)

train_label_data = pd.concat([dfTrainLabels, dfTrain], axis=1)
train_label_data.columns=['Code', 'Blue', 'Green', 'Red', 'NIR']
train_label_data.to_csv('train.csv')

np.save('train.npy', final_data)
# datasetTestGT = gdal.Open('E:/Ceng463/Proje_Gibraltar/S2B_MSIL1C_20220528_Test_GT.tif')

# # Read the data from the first GeoTIFF file into a NumPy array
# testGT2d = datasetTestGT.ReadAsArray()
# testGT2d = testGT2d[1:, :]
# testGT2d = np.swapaxes(testGT2d, 0, 1)
# # Convert the 2-dimensional NumPy arrays into 2-dimensional arrays with rows and columns
# testGT1d = testGT2d.reshape(testGT2d.shape[0] * testGT2d.shape[1], 1)

# # Convert the combined array into a Pandas DataFrame
# df = pd.DataFrame(testGT1d)

# # Export the DataFrame as a CSV file
# df.to_csv('test_gt.csv')
# np.save('test_gt.npy', testGT1d)

datasetTest = gdal.Open(r'C:\Users\muham\Downloads\Project-20230123T143514Z-001\Project/S2B_MSIL1C_20220528_Test.tif')

# Read the data from the first GeoTIFF file into a NumPy array
dataTest2d = datasetTest.ReadAsArray()
dataTest2d = np.transpose(dataTest2d)
# Convert the 2-dimensional NumPy arrays into 2-dimensional arrays with rows and columns
dataTest1d = dataTest2d.reshape(dataTest2d.shape[0] * dataTest2d.shape[1], -1)
np.save('test_all.npy', dataTest1d)
# Convert the combined array into a Pandas DataFrame
dfTest = pd.DataFrame(dataTest1d)
dfTest.columns=['Blue', 'Green', 'Red', 'NIR']
# Export the DataFrame as a CSV file
dfTest.to_csv('test.csv')


mask = dataTraining1d[:,3] != 0

# Use the mask to keep only the non-zero rows in the data and labels
dataTraining1d = dataTraining1d[mask]
trainGT1d = trainGT1d[mask]

#Normalize Data between 0 and 1 before using
from sklearn.model_selection import train_test_split
dataTest1d = dataTest1d.astype(float) / 10000
dataTraining1d = dataTraining1d.astype(float) / 10000



X_Train = dataTraining1d
y_Train = trainGT1d
X_Val, X_Train, y_val, y_Train = train_test_split(dataTraining1d, trainGT1d, stratify=trainGT1d, test_size=0.010)

from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

# Create the KNN classifier with k=1
#clf = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier


#clf = OneVsRestClassifier(GradientBoostingClassifier(n_estimators=100, random_state=42), n_jobs=-1)
# Use cross-validation to evaluate the model's accuracy
# scores = cross_val_score(clf, dataTraining1d, np.ravel(trainGT1d), cv=5)
# acc = scores.mean()

import time

start_time = time.time()

# Fit the classifier to the data
model = DecisionTreeClassifier()

model.fit(X_Train, np.ravel(y_Train))

elapsed_time = time.time() - start_time

print(elapsed_time)

# Predict labels for new data
y_predictions = model.predict(X_Val)

# # # Evaluate the accuracy of the classifier
accuracy = metrics.accuracy_score(y_val, y_predictions)
print(f'Accuracy: {accuracy:.2%}')

# Compute the confusion matrix
labels = ['Tree cover', 'Shrubland', 'Grassland', 'Cropland', 'Built-up', 'Bare/sparse vegetation', 'Snow and ice','Permanent water bodies', 'Herbaceous wetland']

#cm = confusion_matrix(y_val, predictions)
#print(classification_report(y_val, predictions,target_names=labels))
# print(cm)
#cmd = ConfusionMatrixDisplay(cm, display_labels=labels)
#cmd.plot()
predictions = model.predict(dataTest1d)
df = pd.DataFrame(predictions)
df.columns=['Code']
# Export the DataFrame as a CSV file
df.to_csv('C:/Users/muham/Desktop/submission.csv')

# import shutil

# # specify the file to compress
# file_to_compress = 'submission.csv'

# # specify the filename for the compressed file
# compressed_file = 'submission.zip'

# # compress the file
# shutil.make_archive(compressed_file.split('.')[0], 'zip', file_to_compress)