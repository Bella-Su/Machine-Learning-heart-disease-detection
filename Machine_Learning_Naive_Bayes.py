import pandas as pd
from math import sqrt
from math import pi
from math import exp
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------------------
# helper function for calculate the Gaussian probability distribution function for x
# ------------------------------------------------------------------------------

def calculate_probability(x, mean, stdev):
	exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
	return (1 / (sqrt(2 * pi) * stdev)) * exponent

# ------------------------------------------------------------------------------
# data preparing
# ------------------------------------------------------------------------------

##  read csv file
dataset = pd.read_csv('heart.csv')

##  splite features and label
X = dataset.drop('target', axis=1)
y = dataset['target']

##  splite data set into training and test sets (8:2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)#, random_state=42
# print("\nTraining set: ", len(X_train)) #242 training, 61 testing
X_train['target'] = y_train

##  seprate the continous and discrete festures
X_train_discrete = X_train.drop(columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca'])
X_train_continu = X_train.drop(columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal'])

# ------------------------------------------------------------------------------
# Training
# ------------------------------------------------------------------------------

## calculate probability of prior probability
Pc1 = y.value_counts()[1]/y.count()
Pc0 = 1 - Pc1

#### discrete data type features ####

##  seprate class 1 and 0 for X_train discrete data
X_train_discrete_1 = X_train_discrete[X_train_discrete['target'] == 1]
X_train_discrete_0 = X_train_discrete[X_train_discrete['target'] == 0]

##  calculate the number of cases from each types in each discrete features
X_train_discrete_1 = X_train_discrete_1.apply(pd.Series.value_counts)
X_train_discrete_0 = X_train_discrete_0.apply(pd.Series.value_counts)
# print(X_train_discrete_1)
# print(X_train_discrete_0)
#     sex  cp    fbs  restecg  exang  slope  thal  target
# 0  63.0  32  116.0     55.0  116.0    6.0     1     NaN
# 1  70.0  32   17.0     77.0   17.0   40.0     4   133.0
# 2   NaN  57    NaN      1.0    NaN   87.0   106     NaN
# 3   NaN  12    NaN      NaN    NaN    NaN    22     NaN
#     sex  cp   fbs  restecg  exang  slope  thal  target
# 0  20.0  78  95.0     57.0   50.0    9.0   NaN   109.0
# 1  89.0   9  14.0     49.0   59.0   70.0  10.0     NaN
# 2   NaN  17   NaN      3.0    NaN   30.0  32.0     NaN
# 3   NaN   5   NaN      NaN    NaN    NaN  67.0     NaN

#   calculate the number of 1s and 0s in the y_train
y_train_sumaries = y_train.value_counts()
# 1    133
# 0    109

## drop the target columns to prevent out of boundries!!!!!!!!!!!
X_train_discrete_1 = X_train_discrete_1.iloc[: , :-1]
X_train_discrete_0 = X_train_discrete_0.iloc[: , :-1]

##  calculate the probability of cases from each types in each discrete features
X_train_discrete_prob = {'sex':[], 'cp':[], 'fbs':[], 'restecg':[], 'exang':[], 'slope':[], 'thal':[]}
temp = []
for key, value in X_train_discrete_0.iteritems():
    for i in value:
        temp.append(i/y_train_sumaries[0])
        X_train_discrete_prob[key].append(temp)
        temp = []
        #   till this step we are only append for same type from {'sex': [[20/109][89/109]]}


for key, value in X_train_discrete_1.iteritems():
    j=0
    for i in value:
        # append_value(X_train_discrete_prob, key, i/y_train_sumaries[1])
        X_train_discrete_prob[key][j].append(i/y_train_sumaries[1])
        j += 1
        #   we want to update {'sex': [[20/109, 63/133][89/109, 70/133]]}

# print(X_train_discrete_prob)


### continous data type features ####

X_train_continu = X_train_continu.iloc[: , :-1]

##  calculate means and std for each continous feature
X_train_continu_mean = X_train_continu.mean(axis = 0)
X_train_continu_std = X_train_continu.std(axis = 0)

##  normalize the continous features for training data set with F(x) = (x - mean)/std
X_train_continu = X_train_continu.sub(X_train_continu_mean, axis='columns').div(X_train_continu_std, axis='columns')
# print(X_train_continu)
#           age  trestbps      chol   thalach   oldpeak        ca
# 132 -1.353992 -0.615580  0.912143  0.531679 -0.918959 -0.688274
# 202  0.384290  1.167072  0.438618 -1.749956 -0.193386 -0.688274
# 196 -0.919422  1.167072 -0.300082 -0.139390  2.346120 -0.688274
# 75   0.058362  0.275746  0.059797  0.486941  0.350794 -0.688274
# 176  0.601575 -0.793845 -0.319023  0.442203  0.350794  1.330664
# ..        ...       ...       ...       ...       ...       ...
# 188 -0.484851  0.572855 -0.262200  0.576417 -0.374780  0.321195
# 71  -0.376209 -2.160545 -0.375846  0.173775 -0.918959  0.321195
# 106  1.579358  1.761290 -0.243259 -0.855197 -0.828263  0.321195
# 270 -0.919422 -0.615580  0.040856 -0.273604 -0.193386 -0.688274
# 102  0.927503  0.572855 -0.981959  1.292224 -0.918959  1.330664

X_train_continu['target'] = y_train

##  seprate class 1 and 0 for X_train continous data
X_train_continu_1 = X_train_continu[X_train_continu['target'] == 1]
X_train_continu_0 = X_train_continu[X_train_continu['target'] == 0]

##   drop the last row again for calculate mean and std
X_train_continu_1 = X_train_continu_1.iloc[: , :-1]
X_train_continu_0 = X_train_continu_0.iloc[: , :-1]


##  calculate means and std for each continous feature after normaliazed
X_train_continu_1_mean = X_train_continu_1.mean(axis = 0)
X_train_continu_1_std = X_train_continu_1.std(axis = 0)
X_train_continu_0_mean = X_train_continu_0.mean(axis = 0)
X_train_continu_0_std = X_train_continu_0.std(axis = 0)

# ------------------------------------------------------------------------------
# Testing
# ------------------------------------------------------------------------------

predic = []

X_test = X_test.T

for key, value in X_test.iteritems():
    #   prior probability
    prob_in_1 = Pc1
    prob_in_0 = Pc0

    for i in value.index:
        if i == 'sex'or i =='cp' or i =='fbs'or i =='restecg'or i =='exang' or i =='slope' or i =='thal':
            prob_in_1 = prob_in_1 * X_train_discrete_prob.get(i)[int(value[i])][1]
            prob_in_0 = prob_in_0 * X_train_discrete_prob.get(i)[int(value[i])][0]

        elif i == 'age'or i == 'trestbps'or i == 'chol'or i == 'thalach'or i == 'oldpeak'or i == 'ca':
            p1 = calculate_probability(value[i], X_train_continu_1_mean[i], X_train_continu_1_std[i])
            if p1 == 0:
                p1 = 0.0001
            prob_in_1 = prob_in_1 * p1

            p0 = calculate_probability(value[i], X_train_continu_0_mean[i], X_train_continu_0_std[i])
            if p0 == 0:
                p0 = 0.0001
            prob_in_0 = prob_in_0 * p0

    if prob_in_1 > prob_in_0:
        predic.append(1)
    else:
        predic.append(0)   


# ------------------------------------------------------------------------------
# Getting Accuracy, Percision, Recall, F1 score, confusion Matrix
# ------------------------------------------------------------------------------

correct = 0   #correct = true positive + true negative
TP = 0
TN = 0
FP = 0
FN = 0

y_test = y_test.to_numpy()

for i in range(len(y_test)):
    if (y_test[i] == predic[i]):
        if predic[i] == 1:
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

total_case = len(y_test)
print("\nConfusion Matrix:")
print("                  true positive  true nagetive")
print("prediction positive|","{:.0%}". format(TP/total_case),"     |   ", "{:.0%}". format(FP/total_case), "|")
print("prediction negative|","{:.0%}". format(FN/total_case), "     |   ", "{:.0%}". format(TN/total_case), "|")