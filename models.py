import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from statistics import mean
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

### read in data
multiple_patients = True
ID = False
input_file = "data/five_subjects.csv"
output_file = "output/five_subjects_results.csv"

# prepare data
data = pd.read_csv(input_file)
X = data.loc[:, data.columns != "seizure"]
X = X.loc[:, X.columns != "start_time"]
X = X.loc[:, X.columns != "file ID"]
Y = np.asarray(data['seizure'])
feature_names = X.columns.tolist()
print('The number of samples for the non-seizure class is:', Y.shape[0])
print('The number of samples for the seizure class is:', np.sum(Y))

# if multiple patients, one-hot encode patient ID
if multiple_patients:
    X = X.loc[:, X.columns != "subject"] 
    if ID:
        patient = pd.get_dummies(data['subject'], prefix='subject')
        X = pd.concat([X, patient], axis = 1)

# ### preprocessing

def filter_features(X):
    # check zero variance features
    thresholder = VarianceThreshold(threshold=0)
    print("Variables Kept after removing features with 0 variance: ", thresholder.fit_transform(X).shape[1])

    # highly correlated features
    corr = abs(x.corr())
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
    cols = [column for column in upper.columns if any(upper[column] < 0.9)]
    print("Variables Kept after removing features with corr > 0.9: ", len(cols)) 

    # normalize features
    X = preprocessing.normalize(X)
    

# # split into testing and training 
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state=0)
# print('The number of samples for the non-seizure class in training is:', y_train.shape[0])
# print('The number of samples for the seizure class in training is:', np.sum(y_train))
# print('The number of samples for the non-seizure class in testing is:', y_test.shape[0])
# print('The number of samples for the seizure class in testing is:', np.sum(y_test))

### Modeling

SVM = SVC(kernel="rbf", class_weight={1: 100}, random_state = 0)

# cross validation
kf = KFold(n_splits=5)
accuracy, TPR, FPR = [], [], []
for train, test in kf.split(x_train):
    SVM.fit(x_train[train, :], y_train[train])
    pred = SVM.predict(x_train[test])
    tn, fp, fn, tp = confusion_matrix(y_train[test], pred).ravel()
    accuracy.append((tp + tn)/(tn + fp + fn + tp))
    TPR.append(tp / (tp + fn))
    FPR.append(fp / (fp + tn))

# hold our validation
start = time.time()
SVM.fit(X_train, y_train)
pred = SVM.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
test_accuracy = ((tp + tn)/(tn + fp + fn + tp))
test_TPR = (tp / (tp + fn))
test_FPR = (fp / (fp + tn))
end = time.time()
hours, rem = divmod(end-start, 3600)
minutes, seconds = divmod(rem, 60)
print("Kernel SVM\n{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
print(confusion_matrix(y_test, pred))
# record results
result = {
            'model': 'Kernel SVM',
            'cv accuracy': mean(accuracy),
            'cv TPR': mean(TPR),
            'cv FPR': mean(FPR),
            'test accuracy': test_accuracy,
            'test TPR': test_TPR,
            'test FPR': test_FPR
        }

print(result)

KNN = KNeighborsClassifier(2)

# cross validation
kf = KFold(n_splits=5)
accuracy, TPR, FPR = [], [], []
for train, test in kf.split(X_train):
    KNN.fit(X_train[train, :], y_train[train])
    pred = KNN.predict(X_train[test])
    tn, fp, fn, tp = confusion_matrix(y_train[test], pred).ravel()
    accuracy.append((tp + tn)/(tn + fp + fn + tp))
    TPR.append(tp / (tp + fn))
    FPR.append(fp / (fp + tn))

# hold our validation
start = time.time()
KNN.fit(X_train, y_train)
pred = KNN.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
test_accuracy = ((tp + tn)/(tn + fp + fn + tp))
test_TPR = (tp / (tp + fn))
test_FPR = (fp / (fp + tn))
end = time.time()
hours, rem = divmod(end-start, 3600)
minutes, seconds = divmod(rem, 60)
print("KNN K=2\n{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
print(confusion_matrix(y_test, pred))
# record results
result.append({
            'model': 'KNN K=2',
            'cv accuracy': mean(accuracy),
            'cv TPR': mean(TPR),
            'cv FPR': mean(FPR),
            'test accuracy': test_accuracy,
            'test TPR': test_TPR,
            'test FPR': test_FPR
        })

final = pd.DataFrame.from_dict(result)
final.to_csv(os.path.join(output_file), index=False) 
print(final)
print("done")