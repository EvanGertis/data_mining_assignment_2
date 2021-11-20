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

def mc(columnName,training_set):
    print(f'Column Name :{columnName}')
    print(f'Column Contents: {training_set[columnName]}')
    column = training_set[columnName]
    probs = column.value_counts(normalize=True)
    print(f'Probability {probs}')
    messageConveyed = -1*np.sum(np.log2(probs)*probs)
    # print(f'mc {messageConveyed}')
    return messageConveyed

def isNotUnique(s):
    a = s.to_numpy() # s.values (pandas<0.24)
    return not (a[0] == a).all()

def ID3(threshold,g):
    # use the training set to predict the test set.
    # use the Assignment 2--Training set to extract rules and test the quality of the extracted rules against the Assignment 2-- Test set for ID3.
    test_set = pd.read_csv("Assignment 2--Test set for ID3.csv")
    training_set = pd.read_csv("Assignment 2--Training set for ID3.csv")

    print('***********************************')
    print('TRAINING SET')
    print(training_set)
    print('***********************************')


    print('***********************************')
    print('TEST SET')
    print(test_set)
    print('***********************************')

    print(f'test_set: {test_set}')
    print(f'training_set: {training_set}')

    # Step 1- Calculate MC (Message Conveyed) for the given data set in reference to the class attribute
    print(f'Step 1- Calculate MC (Message Conveyed) for the given data set in reference to the class attribute')
    # MC = -p1*log2(p1) - p2*log2(p2)
    # For n classes MC = -p1log2(p1) - p2*log2(p2)-...-pn*log2(pn)

    # For each column calculate the gain.
    numberOfColumns = 0
    mcDictionary = {}
    print('***********************************')
    print('For each column calculate the gain.')
    for (columnName, columnData) in training_set.iteritems():
        messageConveyed = mc(columnName,training_set)
        mcDictionary.update({columnName:round(messageConveyed)})
        numberOfColumns+=1
    print('***********************************')
    print(f'numberOfColumns {numberOfColumns}')
    print(f'mcDictionary {mcDictionary}')
    
    
    # The column with the highest gain is the root.
    print(f'The column with the highest gain is the root.')
    values = mcDictionary.values()
    max_value = max(values)
    print(f'The max value is {max_value}')
    # print(f'The max value, {max_value}, is associated with column {columnWithMaximumInformationGain}')
    val_list = list(values)
    columnWithMaximumInformationGain = list(mcDictionary.keys())[list(mcDictionary.values()).index(max_value)]
    print(f'The max value, {max_value}, is associated with column {columnWithMaximumInformationGain}')

    # select the max value from the gain array
    # this is the new root
    root =  training_set[columnWithMaximumInformationGain]
    print(f'root {root}')   

    # Loop
    # Step 2 - Repeat for every attribute
    print(f'Step 2 - Repeat for every attribute')
    for (columnName, columnData) in training_set.iteritems():

        # # leaf generated from the decision tree.
        # F1 = training_set[columnName].mode()
        # print(f'F1 : {F1} leaf generated from the decision tree.')

        # # # define c1 count of records w/ dominant class in F1
        # # # How do I determine the number of records w/ dominant class in F1?
        # c1 = training_set[columnName].count()

        # # # alpha = c1/ |F1|
        # # # F1 is one of the unique values of a given attribute.
        # alpha = c1/ abs(F1)
        # print(f'alpha {alpha}')
        # exit()
        # # the number of records in the test set that are correctly classified by the rules extracted from the tree before removal.
        # # How do I determine the number of records in test set that are correctly classified by rules extracted from the tree before removal?
        # N = 0

        # # the number of records in the test set that are correctly classified by the rules extracted from the tree.
        # # How do I determine the number of records in the test set that are correctly classified by the rules extracted from the tree?
        # M = 0

        # # the parameter and 0 <= g <= 0.15
        # g = 0

        if g < 0 or g > 0.15:
            exit()

        # k is the total number of branches in the subtree
        # i) use the atttribute as a node from which k 
        # k branches are emanating, where k is
        # the number of unique values in the attribute
        attribute = columnName
        k         = training_set[columnName].nunique()
        print("*****************************************")
        print("**************** i  *********************")
        print(f'use the atttribute {columnName} as a node from which {k}')
        print(f'branches are emanating, where {k} is')
        print(f'the number of unique values in the attribute')

        # if alpha > threshold:
        #     # stop splitting tree

        # ii) split the given data source based on the
        # unique values in the attribute
        print("*****************************************")
        print("**************** ii *********************")
        print(f'split the given data source based on the')
        print(f'unique values in the attribute: {columnName}')
        df1 = training_set[training_set[columnName] >= k]
        df2 = training_set[training_set[columnName] < k]

        print("**********")
        print("splitting ")
        print(f'df1 {df1}')
        print(f'df2 {df2}')
        print("**********")

        # iii) calculate MC for new splits
        # calculate MC for each  attribute of Venue
        print("*****************************************")
        print("************* iii ***********************")
        print(f"calculate MC for new splits")
        print(f"calculate MC for each  attribute of {columnName}")
        messageConveyed = mc(columnName,training_set)
        print(f"MC for {columnName} is {messageConveyed}")

        # iv calculculate the weight for each split
        # start with venue
        print("*****************************************")
        print("************* iv  ***********************") 
        print(f"calculculate the weight for each split ({columnName})")
        # Loop 
        # For each unique value calculate unique_value/total
        uniques1 = df1[columnName].unique()
        uniques2 = df2[columnName].unique()
        total1   = df1[columnName].count()
        total2   = df2[columnName].count()

        print(f'unique values for {columnName} branch 1 is {uniques1}') 
        print(f'unique values for {columnName} branch 2 is {uniques2}')


        print("*****************************************")
        print("*************  v  ***********************") 
        print(f"calculate the weighted MC (WMC) for the attribute ({columnName})")
        print("*****************************************")
        print("************* weights for df1  ***********")
        print(f"WMC({columnName})")
        wmc1 = 0
        for unique_value in uniques1:
            weight = unique_value/total1
            print(f"{weight} = {unique_value}/{total2}")
            wmc1 = weight*mc(columnName,df1)
            print(f"+= {wmc1}")

        # vi) Calculate Gain for the attribute [MC-WMC(venue)]
        # Gain(venue) = MC-WMC(venue)
        print("*****************************************")
        print("*************  vi  **********************") 
        print(f"Calculate Gain for the {columnName} [{messageConveyed-wmc1}]")
        gain = messageConveyed-wmc1
        print(f"wmc1 : {wmc1}")
        print(f"gain for branch 1 of {columnName} is {gain}")

    

        # v) calculate the weighted MC (WMC) for the attribute
        # WMC(venue) = W(1)*MC(1) + W(2)*MC(2)
        print("*****************************************")
        print("*************  v  ***********************") 
        print(f"calculate the weighted MC (WMC) for the attribute ({columnName})")
        print("*****************************************")
        print("************* weights for df2  ***********")
        print(f"WMC({columnName})")
        wmc2 = 0
        for unique_value in uniques2:
            weight = unique_value/total2
            print(f"{weight} = {unique_value}/{total2}")
            messageConveyed = mc(columnName,df2)
            wmc2 += weight*messageConveyed
            print(f"{wmc2} += {weight}*{messageConveyed}")

        # vi) Calculate Gain for the attribute [MC-WMC(venue)]
        # Gain(venue) = MC-WMC(venue)
        print("*****************************************")
        print("*************  vi  **********************") 
        print(f"Calculate Gain for the {columnName} [{messageConveyed-wmc2}]")
        gain = messageConveyed-wmc1
        print(f"wmc2 : {wmc2}")
        print(f"gain for branch 2 of {columnName} is {gain}")
        
        if(columnName == 'Venue'):
            exit()

        # Step 3- Repeat for each split produced by the root
        # if all records have the same class then break. 
        if(isNotUnique(df1[columnName])):
            break

        if(isNotUnique(df2[columnName])):
            break

        # Step 4- If every split is free of a mixture of class values, then stop
        # expansion of the tree

        # # How do we apply prepruning to the data?
        # # For post-pruning use the criteria below
        # if (N-M)/Q < g*k:
        #     # remove subtree

        # Step 5- Extract rules in form of if-then-else from the tree
    
    
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
selection = "ID3" #= input("Please enter your selection for either ID3 or Bayes classification: ")
threshold = 0.9   #= input("Please enter a threshold: ")
g         = 0.05   #= input("Please enter a value for g: ")

if(selection == "ID3"):
    ID3(threshold,g)

if(selection == "Bayes"):
    BayesClassifier()