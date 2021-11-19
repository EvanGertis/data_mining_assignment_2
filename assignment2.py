from numpy.core.defchararray import count
import pandas as pd
import numpy as np
import numpy as np
from math import ceil, floor, log2
from sklearn.decomposition import PCA
from numpy import linalg as LA
from sklearn.tree import DecisionTreeClassifier

def calculate_metrics(tp, tn, fn, p, n, fp):
    # calculate the accuracy, error rate, sensitivity, specificity, and precision for the selected classifier in reference to the corresponding test set.
    accuracy = tp + tn /(p+n)
    error_rate = fp + fn /(p + n)
    sensitivity = tp/ p
    precision = tp/ (tp+fp)
    specificity = tn/n

    display_metrics(accuracy, error_rate, sensitivity, precision, specificity)

def display_metrics(accuracy, error_rate, sensitivity, precision, specificity):
    print(f'Accuracy: {accuracy}, Error_rate:{error_rate}, Sensitivity:{sensitivity}, Precision:{precision}, specificity:{specificity}')

def ID3(threshold,g):
    # use the training set to predict the test set.
    # use the Assignment 2--Training set to extract rules and test the quality of the extracted rules against the Assignment 2-- Test set for ID3.
    test_set = pd.read_csv("Assignment 2--Test set for ID3.csv", header=None)
    training_set = pd.read_csv("Assignment 2--Training set for ID3.csv", header=None)

    print(f'test_set: {test_set}')
    print(f'training_set: {training_set}')

    # Step 1- Calculate MC (Message Conveyed) for the given data set in reference to the class attribute
    # MC = -p1*log2(p1) - p2*log2(p2)
    # For n classes MC = -p1log2(p1) - p2*log2(p2)-...-pn*log2(pn)

    # For each column
    # For each row
    # calculate the probability for an attribute

    for key, value in test_set.iteritems():
        if (isinstance(key, str)):
            print(f"Processing {key}")
        print("********")
        print(value)
        print()
        print("********")
    
    for key, value in test_set.iteritems():
        if (isinstance(key, str)):
            print(f"Processing {key}")
        print("********")
        print(value)
        print()
        print("********")


    # set gain array

    # Loop
        # Step 2 - Repeat for every attribute

        # i) use the atttribute as a node from which k 
        # k branches are emanating, where k is
        # the number of unique values in the attribute

        # ii) split the given data source based on the
        # unique values in the attribute

        # iii) calculate MC for new splits
        # calculate MC for each  attribute of Venue

        # iv calculculate the weight for each split
        # start with venue
        
        # v) calculate the weighted MC (WMC) for the attribute
        # WMC(venue) = W(1)*MC(1) + W(2)*MC(2)

        # vi) Calculate Gain for the attribute [MC-WMC(venue)]
        # Gain(venue) = MC-WMC(venue)

        # Step 3- Repeat for each split produced by the root
        # if all records have the same class then break. 

        # Step 4- If every split is free of a mixture of class values, then stop
        # expansion of the tree

        # Step 5- Extract rules in form of if-then-else from the tree
    
    # select the max value from the gain array
    # this is the new root



    # # leaf generated from the decision tree.
    # F1 = 0

    # # define c1 count of records w/ dominant class in F1
    # # How do I determine the number of records w/ dominant class in F1?
    # c1 = 0

    # # alpha = c1/ |F1|
    # # F1 is one of the unique values of a given attribute.
    # alpha = c1/ abs(F1)

    # # the number of records in the test set that are correctly classified by the rules extracted from the tree before removal.
    # # How do I determine the number of records in test set that are correctly classified by rules extracted from the tree before removal?
    # N = 0

    # # the number of records in the test set that are correctly classified by the rules extracted from the tree.
    # # How do I determine the number of records in the test set that are correctly classified by the rules extracted from the tree?
    # M = 0

    # # the parameter and 0 <= g <= 0.15
    # g = 0

    # if g < 0 or g > 0.15:
    #     exit()

    # # k is the total number of branches in the subtree
    # # How do I determine the total number of branches in the subtree?
    # k = 0

    # if alpha > threshold:
    #     # stop splitting tree

    # # How do we apply prepruning to the data?

    # # For post-pruning use the criteria below
    # if (N-M)/Q < g*k:
    #     # remove subtree
    
    # # true positive
    # tp = 0 
    # # true negative
    # tn = 0
    # # postive
    # p  = 0
    # #  negative
    # n  = 0
    # # false positive
    # fp = 0

    # calculate_metrics(tp, tn, p, n, fp)

def BayesClassifier():
    # use the assignment 2-- training set for Bayes as the training set to classify the records of the assignment 2 test set for bayes
    test_set = pd.read_csv("Assignment 2--Test set for Bayes.csv")
    training_set = pd.read_csv("Assignment 2--Training set for Bayes.csv")


# prompt user to select either ID3 or Bayes classifier.
selection = input("Please enter your selection for either ID3 or Bayes classification: ")
threshold = input("Please enter a threshold: ")
g         = input("Please enter a value for g: ")

if(selection == "ID3"):
    ID3(threshold,g)

if(selection == "Bayes"):
    BayesClassifier()