# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score  

from scipy import optimize as op
data = pd.read_csv('Autism-Adult-Data.arff',na_values="?")
#data = pd.read_csv('autism_child_data.csv',na_values="?")
#data = pd.read_csv('Autism_data.csv',na_values="?")
plt.figure(figsize=(10,7))
sb.heatmap(data.isnull(),cmap="viridis",cbar=False,yticklabels=False)
print(data.head())
data1=data
total_missing_data = data.isnull().sum().sort_values(ascending=False)

percent_of_missing_data = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending=False)

missing_data = pd.concat(
    [
        total_missing_data, 
        percent_of_missing_data
    ], 
    axis=1, 
    keys=['Total', 'Percent']
)
print(missing_data.head())

data.rename(columns={'Class/ASD': 'decision_class'}, inplace=True)
data.jundice = data.jundice.apply(lambda x: 0 if x == 'no' else 1)
data.decision_class = data.decision_class.apply(lambda x: 0 if x == 'NO' else 1)
data.austim = data.austim.apply(lambda x: 0 if x == 'no' else 1)
le = LabelEncoder()
data.gender = le.fit_transform(data.gender) 
data.drop(['result'], axis=1, inplace=True)

print(data.isnull().sum())


X=data[['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score', 'A6_Score',
       'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score', 'age', 'austim', 'gender',
       'jundice']]
print(X)
Y=data[['decision_class']]
print(Y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=101)


def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

def regCostFunction(theta, X, y, _lambda = 0.1):
    m = y.size
    h = sigmoid(X.dot(theta))
    reg = (_lambda/(2 * m)) * np.sum(theta**2)

    return (1 / m) * (-y.T.dot(np.log(h)) - (1 - y).T.dot(np.log(1 - h))) + reg

def regGradient(theta, X, y, _lambda = 0.1):
    m, n = X.shape
    theta = theta.reshape((n, 1))
    y = y.reshape((m, 1))
    h = sigmoid(X.dot(theta))
    reg = _lambda * theta /m

    return ((1 / m) * X.T.dot(h - y)) + reg

def logisticRegression(X, y, theta):
    result = op.minimize(fun = regCostFunction, x0 = theta, args = (X, y),
                         method = 'TNC', jac = regGradient)  
    return result.x
    
Autism_Status = [1, 0]
m = data.shape[0]
n = 13
k = 2
print(Autism_Status)
all_theta = np.zeros((k, n + 1))
i = 0
for Autism in Autism_Status:
    tmp_y = np.array(y_train ==  Autism, dtype = int)
    optTheta = logisticRegression(X_train, tmp_y, np.zeros((n + 1,1)))
    all_theta[i] = optTheta
    i += 1    
P = sigmoid(X_test.dot(all_theta.T)) #probability for each flower
p = [Autism_Status[np.argmax(P.values[i, :])] for i in range(X_test.shape[0])]

print("Test Accuracy ", accuracy_score(y_test, p) * 100 , '%')

print(classification_report(y_true=y_test,y_pred=p))

cfm = confusion_matrix(y_test, p, labels = Autism_Status)
sb.heatmap(cfm, annot = True, xticklabels = Autism_Status, yticklabels = Autism_Status);