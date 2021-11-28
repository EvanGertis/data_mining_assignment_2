from numpy.core.defchararray import count
import pandas as pd
import numpy as np
import math
from sklearn.decomposition import PCA
from numpy import linalg as LA
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

def calculate_metrics(training_set,test_set,classAttribute,classValue):
    # calculate the accuracy, error rate, sensitivity, specificity, and precision for the selected classifier in reference to the corresponding test set.
    tp = len(training_set[training_set[classAttribute] == classValue].index)
    fp = len(test_set[test_set[classAttribute] == classValue].index)
    tn = len(training_set[training_set[classAttribute] == classValue].index) 
    fn = len(test_set[test_set[classAttribute] != classValue.index])
    p  = tp + fp
    n  = tn + fn
    print(f" \t      \t\t {classValue} \t not {classValue} \t \t TOTAL")
    print(f" \t      \t\t  \t  \t \t ")
    print(f" \t      \t {classValue} \t {tp}  \t {fp} \t {p}")
    print(f" \t not  \t {classValue} \t {fn}  \t {tn} \t {n}")
    print(f" \t total\t\t {tp+fn} \t {fn+tn}  \t {p+n} \t")

    accuracy = tp + tn /(p+n)
    error_rate = fp + fn /(p + n)
    sensitivity = tp/ p
    precision = tp/ (tp+fp)
    specificity = tn/n

    display_metrics(accuracy, error_rate, sensitivity, precision, specificity)

def display_metrics(accuracy, error_rate, sensitivity, precision, specificity):
    print(f'Accuracy: {accuracy}, Error_rate:{error_rate}, Sensitivity:{sensitivity}, Precision:{precision}, specificity:{specificity}')

# Step 1- Calculate MC (Message Conveyed) for the given dataset (let us call it file TF) in reference to  the class attribute 
# MC(TF) = -p1*log2(p1) - p2*log2(p2) 
def mc(classAttribute,attribute,training_set):
    column = training_set[classAttribute]

    if attribute:
        column = training_set[training_set[classAttribute] == attribute] 

    probs = column.value_counts(normalize=True)
    messageConveyed = -1*np.sum(np.log2(probs)*probs)
    return messageConveyed

def wmc(classAttribute,attribute,training_set):
    attributeCount = len(training_set[training_set[classAttribute] == attribute].index)
    total          = len(training_set[classAttribute].index)
    return attributeCount/total

def ID3(root,training_set,test_set, threshold, g):

    highestGainAttribute = ""
    highestGainValue     = -math.inf
    for classAttribute, values in training_set.iteritems():
        messageConveyed = mc(classAttribute, attribute=None, training_set=training_set)

        attributes = training_set[classAttribute].unique()
        weightedMessageConveyed = 0
        for attribute in attributes:
            weight = wmc(classAttribute, attribute, training_set)
            messageConveyed = mc(classAttribute, attribute, training_set)
            weightedMessageConveyed += weight*messageConveyed

        gain = messageConveyed - weightedMessageConveyed
        if gain > highestGainValue:
            highestGainAttribute = classAttribute
            highestGainValue     = gain
    
    root = highestGainAttribute
    leaves = training_set[root].unique()
    splits = {}
    print(f"root {root}")
    for leaf in  leaves:
        if training_set[training_set[root] == leaf]["Volume"].is_unique:
            splits.update({leaf:"no split"})
            return
        else:
            splits.update({leaf:"split"})
    classValues = None
    for leaf,split in splits.items():
        if root in training_set:
            c1 = len(training_set[training_set[root] == leaf].index)
            F1 = len(training_set[root].index)
            alpha = c1/F1
            print(f"leaf :{leaf} -> ")
            if split == "split" and alpha < threshold:
                training_set = training_set[training_set[root] == leaf].drop(columns=root)
                ID3(root,training_set,test_set,threshold,g)
            else:
                print(training_set)
                print("end")

    

# use the training set to predict the test set.
# use the Assignment 2--Training set to extract rules and test the quality of the extracted rules against the Assignment 2-- Test set for ID3.
test_set_ID3 = pd.read_csv("Assignment 2--Test set for ID3.csv")
training_set_ID3 = pd.read_csv("Assignment 2--Training set for ID3.csv")


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

    calculate_metrics(training_set,test_set,classAttribute,classWithMaxValue)

# prompt user to select either ID3 or Bayes classifier.
selection = "Bayes" #= input("Please enter your selection for either ID3 or Bayes classification: ")
threshold = 0.5   #= input("Please enter a threshold: ")
g         = 0.01   #= input("Please enter a value for g: ")

root = ""
if(selection == "ID3"):
    if g < 0 or g >=0.015:
        print("g must be between 0<g<0.015")
        exit()
    ID3(root,training_set_ID3,test_set_ID3, threshold, g)

if(selection == "Bayes"):
    BayesClassifier(training_set_Bayes,test_set_Bayes)

