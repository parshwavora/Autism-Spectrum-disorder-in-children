# -*- coding: utf-8 -*-


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score  
from sklearn.neighbors import KNeighborsClassifier  
#data = pd.read_csv('Autism-Adult-Data.arff',na_values="?")
#data = pd.read_csv('autism_child_data.csv',na_values="?")
data = pd.read_csv('Autism_data.csv',na_values="?")
plt.figure(figsize=(10,7))
sb.heatmap(data.isnull(),cmap="viridis",cbar=False,yticklabels=False)
print(data.head())

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
y_train=y_train.values.ravel()

classifier = KNeighborsClassifier(n_neighbors=11)  
classifier.fit(X_train, y_train) 
y_pred = classifier.predict(X_test)    
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred)) 
print("Accuracy is ", accuracy_score(y_test,y_pred)*100)
Autism_Status = [1, 0]
cfm = confusion_matrix(y_test, y_pred, labels = Autism_Status)
sb.heatmap(cfm, annot = True, xticklabels = Autism_Status, yticklabels = Autism_Status);

k_range = range(1, 26)

scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))

print(scores)
plt.plot(k_range, scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')
plt.show() 

