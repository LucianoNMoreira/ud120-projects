#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

from sklearn import svm
from sklearn.metrics import accuracy_score

# clf = svm.SVC(kernel="linear")
# clf = svm.LinearSVC()
clf = svm.SVC(C=10000.0, kernel="rbf")

#Diminuir o dataset para 1%
# features_train = features_train[:len(features_train)/100] 
# labels_train = labels_train[:len(labels_train)/100] 

t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

t0 = time()
predicts = clf.predict(features_test)
print "predict time:", round(time()-t0, 3), "s"

print "Prediction 10 = " + str(predicts[10])
print "Prediction 26 = " + str(predicts[26])
print "Prediction 50 = " + str(predicts[50])

chris = 0
for i in range(len(predicts)):
    if(predicts[i] == 1):
        chris+= 1

print "Chris e-mails: " + str(chris)

# print predicts
t0 = time()
print "Accuracy: %f" % accuracy_score(labels_test, predicts)
print "accuracy time:", round(time()-t0, 3), "s"


#########################################################


