
from osgeo import gdal
import numpy as np
import pandas as pd

datasetTrainingGT = gdal.Open(r'C:\Users\muham\Downloads\Project-20230123T143514Z-001\Project\S2A_MSIL1C_20220516_Train_GT.tif')

trainGT2d = datasetTrainingGT.ReadAsArray()
trainGT2d = np.transpose(trainGT2d)
trainGT1d = trainGT2d.reshape(-1, 1)

dfTrainLabels = pd.DataFrame(trainGT1d)

np.save('train_gt.npy', trainGT1d)

datasetTraining = gdal.Open(r'C:\Users\muham\Downloads\Project-20230123T143514Z-001\Project\S2A_MSIL1C_20220516_TrainingData.tif')

dataTraing = datasetTraining.ReadAsArray()
dataTraing = np.transpose(dataTraing)
dataTraining1d = dataTraing.reshape(dataTraing.shape[0] * dataTraing.shape[1], -1)
dfTrain = pd.DataFrame(dataTraining1d)

final_data = np.concatenate([trainGT1d, dataTraining1d], axis=1)

train_label_data = pd.concat([dfTrainLabels, dfTrain], axis=1)
train_label_data.columns=['Code', 'Blue', 'Green', 'Red', 'NIR']
train_label_data.to_csv('train.csv')

np.save('train.npy', final_data)

datasetTest = gdal.Open(r'C:\Users\muham\Downloads\Project-20230123T143514Z-001\Project/S2B_MSIL1C_20220528_Test.tif')

dataTest2d = datasetTest.ReadAsArray()
dataTest2d = np.transpose(dataTest2d)
dataTest1d = dataTest2d.reshape(dataTest2d.shape[0] * dataTest2d.shape[1], -1)
np.save('test_all.npy', dataTest1d)
dfTest = pd.DataFrame(dataTest1d)
dfTest.columns=['Blue', 'Green', 'Red', 'NIR']
dfTest.to_csv('test.csv')


mask = dataTraining1d[:,3] != 0

dataTraining1d = dataTraining1d[mask]
trainGT1d = trainGT1d[mask]

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

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression

import time

start_time = time.time()

model = OneVsRestClassifier(SVC())

model.fit(X_Train, np.ravel(y_Train))

elapsed_time = time.time() - start_time

print(elapsed_time)

y_predictions = model.predict(X_Val)

accuracy = metrics.accuracy_score(y_val, y_predictions)
print(f'Accuracy: {accuracy:.2%}')

labels = ['Tree cover', 'Shrubland', 'Grassland', 'Cropland', 'Built-up', 'Bare/sparse vegetation', 'Snow and ice','Permanent water bodies', 'Herbaceous wetland']

predictions = model.predict(dataTest1d)
df = pd.DataFrame(predictions)
df.columns=['Code']
df.to_csv('C:/Users/muham/Desktop/submission.csv')

