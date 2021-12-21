import numpy as np
from pandas import read_csv
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer

# ------------------------------------------------------------------------------
# helper function for calculate euclidean distance
# ------------------------------------------------------------------------------

def euclidean_distance(x1,x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# ------------------------------------------------------------------------------
# data preparing
# ------------------------------------------------------------------------------

##  read csv file
dataset = read_csv('heart.csv')

##  splite features and label
X = dataset.drop('target', axis=1)
y = dataset['target']

# apply one hot-hot encoding to categorical features
X_trans = make_column_transformer((OneHotEncoder(),['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']), remainder='passthrough')
X = X_trans.fit_transform(X)

##  convert to np array for easy calculation later
X = np.array(X)
y = np.array(y)

# [[  0.    1.    0.  ... 150.    2.3   0. ]
#  [  0.    1.    0.  ... 187.    3.5   0. ]
#  [  1.    0.    0.  ... 172.    1.4   0. ]
#  ...
#  [  0.    1.    1.  ... 141.    3.4   2. ]
#  [  0.    1.    1.  ... 115.    1.2   1. ]
#  [  1.    0.    0.  ... 174.    0.    1. ]]
#   shap of X: (303, 26)
#   shap of y: (303,)


# in order to get test, valid, test 3 datasets
# firstly, split the data in training set and remaining dataset
X_train, X_rem, y_train, y_rem = train_test_split(X,y, train_size=0.6)

# secondly, split the remaining dataset into validation set and test set
test_size = 0.5
X_valid, X_test, y_valid, y_test = train_test_split(X_rem,y_rem, test_size=0.5)

# ------------------------------------------------------------------------------
# tuning process: find the best k value
# ------------------------------------------------------------------------------

##  create reusable functions for both tuning and testing process
class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self.helper_predict(x) for x in X]
        return np.array(y_pred)

    def helper_predict(self, x):
        ## Compute distances between x and all examples in the training set
        ## x will be from either X_valid or X_test set
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        ## using argsort() to Sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[: self.k]

        ## getting labels of the k nearest neighbor from training set
        k_neighbor_labels = [self.y_train[i] for i in k_indices]

        ## take a vote, return only the most common class label
        ## most_common(1) return a tuple (label, number of label)
        most_common = Counter(k_neighbor_labels).most_common(1)

        ## return the label
        return most_common[0][0]

##  initial the best k
best_k = 1
best_acc = 0
for i in range(1, 10, 2):
    clf = KNN(k=i)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_valid)
    accuracy = np.sum(y_valid == predictions) / len(y_valid) *100.0
    
    if accuracy > best_acc:
        best_acc = accuracy
        best_k = i

    print("Accuracy for k=", i, ": ", accuracy, "%")

print("\nThe best k is ", best_k)
print("----------------------------------------")

# ------------------------------------------------------------------------------
# Testing and getting accuracy
# ------------------------------------------------------------------------------

clf = KNN(k=best_k)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

correct = 0   #correct = true positive + true negative
TP = 0
TN = 0
FP = 0
FN = 0

for i in range(len(y_test)):
    if (y_test[i] == predictions[i]):
        if predictions[i] == 1:
            TP += 1
        else:
            TN += 1
        correct += 1
    else:
        if(y_test[i] == 1):
            FN += 1
        else:
            FP += 1

accuracy=(correct/float(len(y_test)))
print("\nAccuracy: ",accuracy*100.0,"%")

precision = TP / (FP + TP)
print("\nPercision: ",precision*100.0,"%")

recall = TP / (TP + FN)
print("\nRecall: ",recall*100.0,"%")

print("\nF1 Score: ", end=" ")
f_score = 2*((precision*recall)/(precision+recall))
print(f_score)

total_case = len(X_test)
print("\nConfusion Matrix:")
print("                  true positive  true nagetive")
print("prediction positive|","{:.0%}". format(TP/total_case),"     |   ", "{:.0%}". format(FP/total_case), "|")
print("prediction negative|","{:.0%}". format(FN/total_case), "     |   ", "{:.0%}". format(TN/total_case), "|")