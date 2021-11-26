from numpy.core.defchararray import count
import pandas as pd
import numpy as np
import numpy as np
from math import ceil, floor, log2
from sklearn.decomposition import PCA
from numpy import linalg as LA
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


# Step 1- Calculate MC (Message Conveyed) for the given dataset (let us call it file TF) in reference to  the class attribute 
# MC(TF) = -p1*log2(p1) - p2*log2(p2) 
def mc(classAttribute,training_set):
    column = training_set[classAttribute]
    probs = column.value_counts(normalize=True)
    messageConveyed = -1*np.sum(np.log2(probs)*probs)
    return messageConveyed

def wmc(classAttribute,attribute,training_set):
    attributeCount = training_set[training_set[classAttribute] == attribute].count()
    total          = training_set[classAttribute].count()
    print(f'{attributeCount}/{total}')
    return attributeCount/total


def ID3(root,training_set,test_set):
    for classAttribute, values in training_set.iteritems():
        messageConveyed = mc(classAttribute, training_set)
        print(f"{classAttribute} mc: {messageConveyed}")

        attributes = training_set[classAttribute].unique()
        print(f"{classAttribute}\n")
        for attribute in attributes:
            weight = wmc(classAttribute, attribute, training_set)
            print(f"wmc({attribute}) = {weight}")

# For n classes MC(TF) = -p1log2(p1) - p2*log2(p2)-...-pn*log2(pn) 
# The probabilities are approximated by relative frequencies. 
# Step 2- Calculate Gain for every attribute in the training set . 
# Loop 1:  
 # For each attribute (Aj) Do: 
# Consider the attribute is a node from which k branches are emanating,  
# where k is the number of unique values in the attribute  
# Temporarily, split the file TF into K new files based on the unique values in the  attribute Aj. 
# Let us call these new files F1, . . ., Fk  
# Total =0; 
# Loop 2 
 # for each new file Fi Do: 
# Calculate MC for the file and call it MC(Fi). 
# Calculate weight for file Fi and call it Weight(Fi) 
# Weight(Fi) = |Fi|/|TF| 
# Calculate the weighted MC (WMC) for file Fi 
# WMC(Fi) = Weight(Fi) * MC(Fi) 
# Total = Total + MC(Fi)  
# End of loop 2 
# Calculate Gain of Aj 
# Gain(Aj) = MC(TF) – Total; 
# End of Loop 1 
# The attribute with the highest gain is the winner. 
# Permanently split the file TF into K new files based on the K unique values of the winner  attribute. 
# Remove the winner attribute from all new K files. 
# Now you have the root of the tree (the winner attribute) and this tree has k leaves, and  each leaf has its own dataset.  
# Step 3- Examine dataset of each leaf.  
# If the attribute class has the same value for all the records in the leaf’s dataset,  then mark the leaf as “no split”  
# else mark it as “split”.  
# Step 4- For each leaf’s dataset that is marked “Split” Do. 
# The dataset become the new TF  
# TF = leaf’s dataset 
# Go to Step 1;  

# use the training set to predict the test set.
# use the Assignment 2--Training set to extract rules and test the quality of the extracted rules against the Assignment 2-- Test set for ID3.
test_set_ID3 = pd.read_csv("Assignment 2--Test set for ID3.csv")
training_set_ID3 = pd.read_csv("Assignment 2--Training set for ID3.csv")

# prompt user to select either ID3 or Bayes classifier.
selection = "ID3" #= input("Please enter your selection for either ID3 or Bayes classification: ")
threshold = 0.9   #= input("Please enter a threshold: ")
g         = 0.05   #= input("Please enter a value for g: ")

root = ""
if(selection == "ID3"):
    print('***********************************')
    print('TRAINING SET')
    print(training_set_ID3)
    print('***********************************')
    
    print('***********************************')
    print('TEST SET')
    print(test_set_ID3)
    print('***********************************')
    ID3(root,training_set_ID3,test_set_ID3)