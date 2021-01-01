# -*- coding: utf-8 -*-

import pandas as pd
from os import system
import seaborn as sb
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score  
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
data = pd.read_csv('Autism-Adult-Data.arff',na_values="?")
#data = pd.read_csv('autism_child_data.csv',na_values="?")
#data = pd.read_csv('Autism_data.csv',na_values="?") 
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

#print(data.isnull().sum())


X=data[['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score', 'A6_Score',
       'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score', 'age', 'austim', 'gender',
       'jundice']]
print(X)
Y=data[['decision_class']]
print(Y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=101)
feature_names=X.columns
print(feature_names)

clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 42,max_depth=3, min_samples_leaf=5)
clf_entropy.fit(X_train, y_train)
y_pred = clf_entropy.predict(X_test)
print(y_pred)
print("Accuracy is ", accuracy_score(y_test,y_pred)*100)
print(classification_report(y_test, y_pred)) 


Autism_Status = [1, 0]
cfm = confusion_matrix(y_test, y_pred, labels = Autism_Status)
sb.heatmap(cfm, annot = True, xticklabels = Autism_Status, yticklabels = Autism_Status);
feature_target = ['1','0']
tree.export_graphviz(clf_entropy, out_file='tree.dot', feature_names = feature_names, class_names = feature_target, filled = True, rounded = True, special_characters = True)
system("dot -Tpng tree.dot -o child_data2.png ")