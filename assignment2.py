# prompt user to select either ID3 or Bayes classifier.

# use the training set to predict the test set.

# use the Assignment 2--Training set to extract rules and test the quality of the extracted rules against the Assignment 2-- Test set for ID3.

# define c1 count of records w/ dominant class in F1

# alpha = c1/ |F1|

# the number of records in the test set that are correctly classified by the rules extracted from the tree before removal.
n

# the number of records in the test set that are correctly classified by the rules extracted from the tree 
m




# the parameter and 0 <= g <= 0.15

# k is the total number of branches in the subtree

if alpha < threshold:
    # stop splitting tree

if (N-M)/Q < gK:
    # remove subtree

# use the assignment 2-- training set for Bayes as the training set to classify the records of the assignment 2 test set for bayes

# calculate the accuracy

accuracy = tp + tn /(p+n)

error_rate = fp + fn /(p + n)

sensitivity = tp/ p

precision = tp/ (tp+fp)

specificity = tn/n