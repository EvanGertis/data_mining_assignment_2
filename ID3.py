#import required libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

#Loading the iris data
data = load_iris()
print('Classes to predict: ', data.target_names)

#extracting data attributes
X = data.data

#extracting target/ class labels
y = data.target

print('number of examples in the data: ', X.shape[0])

#using the train_test_split to create and 
#train test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=47, test_size=0.25)

#importing the decision tree classifier from sklearn
#library
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion='entropy')

#training the decision tree classifier
clf.fit(X_train, y_train)

#Predicting labels on the test set.
y_pred = clf.predict(X_test)

#importing the accuracy metrics from 
#sklearn.metrics library
from sklearn.metrics import accuracy_score
print('Accuracy Score ontrain data: ', accuracy_score(y_true=y_train, y_pred=clf.predict(X_train)))

print('Accuuracy Score on test data: ', accuracy_score(y_true=y_test, y_pred=y_pred))
