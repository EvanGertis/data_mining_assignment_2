from numpy.core.defchararray import count, index
import pandas as pd
import numpy as np
import math
from sklearn.decomposition import PCA
from numpy import linalg as LA
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

test_set_Bayes = pd.read_csv("Assignment 2--Training set for Bayes.csv")
training_set_Bayes = pd.read_csv("Assignment 2--Test set for Bayes.csv")

def prob_continous_value(A, v, classAttribute, dataset, x):
    # calcuate the average for all values of A in dataset with class = x
    a = dataset[dataset[classAttribute] == x][A].mean()
    # calculate the standard deviation for all values A in dataset with class = x
    stdev = 1
    stdev = dataset[dataset[classAttribute] == x][A].std()
    v = dataset[A].iloc[0]
    if stdev == 0.0:
        stdev = 0.00000000000001
    return (1/(math.sqrt(2*math.pi)*stdev))*math.exp(-((v-a)*(v-a))/(2*stdev*stdev))

def BayesClassifier(training_set,test_set):
    classAttribute = 'Volume'
    products = []
    max = -math.inf
    classWithMaxValue = "" 
    for x in training_set[classAttribute].unique():
        D = len(training_set[classAttribute].index)
        d = len(training_set[training_set[classAttribute] == x].index)
        pClassAttribute = d/D
        print("********")
        print(f'Step 1 calculate p({classAttribute}={x})={pClassAttribute}')
        p = 0
        probabilitiesProduct = 1
        print("********")
        print("Step 2 calculate product of probabilities")
        for A, values in training_set.iteritems():
            if not A == classAttribute:
                v = training_set[A].iloc[0]
                p = prob_continous_value(A, v, classAttribute, training_set, x)
                print(f'p({A}={v}|{classAttribute}={x})={p}')
                probabilitiesProduct *= p
        print(f"probabilitiesProduct={probabilitiesProduct}")
        print("********")
        # products.append(probabilitiesProduct)
        ptotal = pClassAttribute*probabilitiesProduct
        print(f'p({classAttribute}={x}|x)={ptotal}')
        if ptotal > max:
            max = ptotal
            classWithMaxValue = x
        print(f"winner is {classAttribute}={classWithMaxValue}")




# prompt user to select either ID3 or Bayes classifier.
selection = "Bayes" #= input("Please enter your selection for either ID3 or Bayes classification: ")


if(selection == "Bayes"):
    BayesClassifier(training_set_Bayes,test_set_Bayes)