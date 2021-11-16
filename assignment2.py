# prompt user to select either ID3 or Bayes classifier.

# use the training set to predict the test set.

# use the Assignment 2--Training set to extract rules and test the quality of the extracted rules against the Assignment 2-- Test set for ID3.

# leaf generated from the decision tree.
F1 = 0

# define c1 count of records w/ dominant class in F1
# How do I determine the number of records w/ dominant class in F1?
c1 = 0

# alpha = c1/ |F1|
# F1 is one of the unique values of a given attribute.
alpha = c1/ abs(F1)

# the number of records in the test set that are correctly classified by the rules extracted from the tree before removal.
# How do I determine the number of records in test set that are correctly classified by rules extracted from the tree before removal?
N = 0

# the number of records in the test set that are correctly classified by the rules extracted from the tree.
# How do I determine the number of records in the test set that are correctly classified by the rules extracted from the tree?
M = 0

# the parameter and 0 <= g <= 0.15
g = 0

if g < 0 or g > 0.15:
    exit()

# k is the total number of branches in the subtree
# How do I determine the total number of branches in the subtree?
k = 0

if alpha > threshold:
    # stop splitting tree

# How do we apply prepruning to the data?

# For post-pruning use the criteria below
if (N-M)/Q < g*k:
    # remove subtree

# true positive
tp = 0 
# true negative
tn = 0
# postive
p  = 0
#  negative
n  = 0
# false positive
fp = 0

# use the assignment 2-- training set for Bayes as the training set to classify the records of the assignment 2 test set for bayes



# calculate the accuracy, error rate, sensitivity, specificity, and precision for the selected classifier in reference to the corresponding test set.

accuracy = tp + tn /(p+n)

error_rate = fp + fn /(p + n)

sensitivity = tp/ p

precision = tp/ (tp+fp)

specificity = tn/n