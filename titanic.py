import pandas as pd
from pandas import DataFrame
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn import svm, metrics
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
#import xgboost as xgb

def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""


data = pd.read_csv("D:/rawDataFiles/titanic_train.csv")
question = pd.read_csv("D:/rawDataFiles/titanic_test.csv")
data['Title'] = data['Name'].apply(get_title)
question['Title'] = question['Name'].apply(get_title)

data['HasCabin'] = data['Cabin'].apply(lambda x: 0 if type(x) == float else 1)
question['HasCabin'] = question['Cabin'].apply(lambda x: 0 if type(x) == float else 1)

data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
question['FamilySize'] = question['SibSp'] + question['Parch'] + 1

data['Solo'] = data['FamilySize'].apply(lambda x: 1 if x==1 else 0)
question['Solo'] = question['FamilySize'].apply(lambda x: 1 if x==1 else 0)

data['Embarked'] = data['Embarked'].fillna('S')
question['Embarked'] = question['Embarked'].fillna('S')
data['Embarked'] = data['Embarked'].apply(lambda x: 1 if x=='C' else x).apply(lambda x: 2 if x=='Q' else x).apply(lambda x: 3 if x=='S' else x)
question['Embarked'] = question['Embarked'].apply(lambda x: 1 if x=='C' else x).apply(lambda x: 2 if x=='Q' else x).apply(lambda x: 3 if x=='S' else x)

data['Fare'] = data['Fare'].fillna(data['Fare'].median())
question['Fare'] = question['Fare'].fillna(data['Fare'].median())

data['Sex'] = data['Sex'].apply(lambda x: 1 if x=='male' else 0)
question['Sex'] = question['Sex'].apply(lambda x: 1 if x=='male' else 0)

title_list = data.Title.unique()
title_age_avg = []
for title in title_list :
    title_age_avg.append(title + " : " + str(data[data.Title == title]['Age'].mean()))
title_age_dic = {'Mr':4, 'Mrs':4, 'Miss':3, 'Master':1, 'Don':4, 'Dona':4, 'Rev':4, 'Dr':4, 'Mme':3, 'Ms':3, 'Major':5, 'Lady':5, 'Sir':5, 'Mlle':3, 'Col':5, 'Capt':6, 'Countess':4, 'Jonkheer':4}
data['TitleEncoding'] = data['Title'].apply(lambda x: title_age_dic[x])
question['TitleEncoding'] = question['Title'].apply(lambda x: title_age_dic[x])

print("### FILL IN NA IN AGE ###")
age_col = ['Age','Pclass','SibSp','FamilySize','TitleEncoding','Solo']
age_data = data[age_col].dropna()
y = age_data['Age']
X = age_data[age_col[1:]]
scaler = MinMaxScaler()
scaler.fit(X)
X= DataFrame(scaler.transform(X))

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1)

ageNN = MLPRegressor(
    hidden_layer_sizes=(10,9,8),
    activation='relu',
    alpha=0.1,
    max_iter=10000,
    random_state=1
)
ageNN.fit(X_train,y_train)
print('train R^2 : ' + str(ageNN.score(X_train, y_train)))
print('test R^2 : ' + str(ageNN.score(X_test, y_test)))
ageNN.fit(X,y)
nanlist = []
for i in range(891):
    if np.isnan(data.Age[i]):
        nanlist.append(i)
X=[]
for i in nanlist :
    X.append([data.Pclass[i], data.SibSp[i], data.FamilySize[i], data.TitleEncoding[i], data.Solo[i]])
X= DataFrame(scaler.fit(X).transform(X))
Age_pred_result = ageNN.predict(X)
c=0

for i in range(891):
   if np.isnan(data.Age[i]):
       data.Age[i] = Age_pred_result[c]
       c+=1

nanlist = []
for i in range(418):
    if np.isnan(question.Age[i]):
        nanlist.append(i)
X=[]
for i in nanlist :
    X.append([question.Pclass[i], question.SibSp[i], question.FamilySize[i], question.TitleEncoding[i], question.Solo[i]])
X= DataFrame(scaler.fit(X).transform(X))
Age_pred_result = ageNN.predict(X)
c=0
for i in range(418):
   if np.isnan(question.Age[i]):
       question.Age[i] = Age_pred_result[c]
       c+=1
print("###########################################")




columns_to_use = ['Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','HasCabin','FamilySize','TitleEncoding','Solo']
data_nontree = data.copy()[columns_to_use]
question_nontree = question.copy()[columns_to_use[1:]]
data_tree = data.copy()[columns_to_use]
question_tree = question.copy()[columns_to_use[1:]]

#NONTREE
data_nontree_X = data_nontree[columns_to_use[1:]]
data_nontree_y = data_nontree['Survived']
nt_train_X, nt_test_X, nt_train_y, nt_test_y = train_test_split(data_nontree_X, data_nontree_y,random_state=1)
scaler.fit(nt_train_X[columns_to_use[1:]])
nt_train_X = DataFrame(scaler.transform(nt_train_X))
nt_test_X = DataFrame(scaler.transform(nt_test_X))

### ANN
ANN = MLPClassifier(
    hidden_layer_sizes=(10,9),
    activation='relu',
    alpha=0.01,
    max_iter=100000,
    random_state=0
)
ANN.fit(nt_train_X,nt_train_y)
print("$$ ANN $$")
print('train accuracy : ' + str(ANN.score(nt_train_X, nt_train_y)))
print('test accuracy : ' + str(ANN.score(nt_test_X, nt_test_y)))

fpr, tpr, thresholds = metrics.roc_curve(nt_train_y,ANN.predict(nt_train_X),pos_label=2)
metrics.auc(fpr,tpr)

### SVM
SVM = svm.SVC()
SVM.fit(nt_train_X,nt_train_y)
print("$$ SVM $$")
print('train accuracy : ' + str(SVM.score(nt_train_X, nt_train_y)))
print('test accuracy : ' + str(SVM.score(nt_test_X, nt_test_y)))

### LOGISTIC REGRESSION
GLM = LogisticRegression()
GLM.fit(nt_train_X,nt_train_y)
print("$$ LOG $$")
print('train accuracy : ' + str(GLM.score(nt_train_X, nt_train_y)))
print('test accuracy : ' + str(GLM.score(nt_test_X, nt_test_y)))
scaler.fit(data[columns_to_use[1:]])
data_nontree_X = DataFrame(scaler.transform(data_nontree_X))
question_nontree = scaler.transform(question_nontree)
submission = pd.DataFrame({'PassengerId': question['PassengerId'], 'Survived':GLM.predict(question_nontree)})
submission.to_csv("C:/Users/Froilan/Desktop/Repository/kaggleResultCSV/python_logistic.csv",index=False)

#TREE
data_tree_X = data_tree[columns_to_use[1:]]
data_tree_y = data_tree['Survived']
t_train_X, t_test_X, t_train_y, t_test_y = train_test_split(data_tree_X, data_tree_y, random_state=1)

#RANDOMFOREST
RF = RandomForestClassifier(
    n_estimators=1000,
    max_depth=4,
    random_state=0,
    n_jobs=-1
)
RF.fit(t_train_X, t_train_y)
print("$$ RF $$")
print('train accuracy : ' + str(RF.score(t_train_X, t_train_y)))
print('test accuracy : ' + str(RF.score(t_test_X, t_test_y)))

#EXTRATREES
ET = ExtraTreesClassifier(
    n_estimators=6000,
    max_depth=3,
    max_features=6,
    random_state=1,
    n_jobs=-1
)
ET.fit(t_train_X, t_train_y)
print("$$ ET $$")
print('train accuracy : ' + str(ET.score(t_train_X, t_train_y)))
print('test accuracy : ' + str(ET.score(t_test_X, t_test_y)))

#GBDT
GBDT = GradientBoostingClassifier(
    learning_rate=0.005,
    n_estimators=100,
    max_depth=4,
    random_state=1
)
GBDT.fit(t_train_X, t_train_y)
print("$$ GBDT $$")
print('train accuracy : ' + str(GBDT.score(t_train_X, t_train_y)))
print('test accuracy : ' + str(GBDT.score(t_test_X, t_test_y)))

#ADABOOST
ADA = AdaBoostClassifier(
    learning_rate=0.01,
    n_estimators=300,
    random_state=0
)
ADA.fit(t_train_X, t_train_y)
print("$$ ADA $$")
print('train accuracy : ' + str(ADA.score(t_train_X, t_train_y)))
print('test accuracy : ' + str(ADA.score(t_test_X, t_test_y)))

#XGB
#XGB = xgb.XGBClassifier(
#    learning_rate=0.01,
#    n_estimators=100,
#    max_depth=4,
#    gamma=0.9,
#    nthread=-1
#)
#XGB.fit(t_train_X, t_train_y)
#print("$$ XGB $$")
#print('train accuracy : ' + str(XGB.score(t_train_X, t_train_y)))
#print('test accuracy : ' + str(XGB.score(t_test_X, t_test_y)))

#WE WILL SELECT ANN, SVM, LOG, ADA

result = DataFrame({'ANN':ANN.predict(nt_train_X),'SVM':SVM.predict(nt_train_X),'LOG':GLM.predict(nt_train_X),'ADA':ADA.predict(t_train_X),'Actual':t_train_y})
x=['ADA','ANN','LOG','SVM']
X_r_train, X_r_test, y_r_train, y_r_test = train_test_split(result[x],result['Actual'],random_state=1)

FINnn = MLPClassifier(
    hidden_layer_sizes=(10,9),
    activation='relu',
    alpha=0.01,
    max_iter=100000,
    random_state=0
)
FINnn.fit(X_r_train,y_r_train)
print("$$ FINnn $$")
print('train accuracy : ' + str(FINnn.score(X_r_train,y_r_train)))
print('test accuracy : ' + str(FINnn.score(X_r_test,y_r_test)))

FINlog = LogisticRegression()
FINlog.fit(X_r_train,y_r_train)
print("$$ FINlog $$")
print('train accuracy : ' + str(FINlog.score(X_r_train,y_r_train)))
print('test accuracy : ' + str(FINlog.score(X_r_test,y_r_test)))

#########################################################################

scaler.fit(data[columns_to_use[1:]])
data_nontree_X = DataFrame(scaler.transform(data_nontree_X))
question_nontree = scaler.transform(question_nontree)

ANN = MLPClassifier(
    hidden_layer_sizes=(10,9),
    activation='relu',
    alpha=0.01,
    max_iter=100000,
    random_state=1
)
ANN.fit(data_nontree_X,data_nontree_y)

SVM = svm.SVC()
SVM.fit(data_nontree_X,data_nontree_y)

GLM = LogisticRegression()
GLM.fit(data_nontree_X,data_nontree_y)

ADA = AdaBoostClassifier(
    learning_rate=0.01,
    n_estimators=300,
    random_state=0
)
ADA.fit(data_tree_X,data_tree_y)

result = DataFrame({'ANN':ANN.predict(data_nontree_X),'SVM':SVM.predict(data_nontree_X),'LOG':GLM.predict(data_nontree_X),'ADA':ADA.predict(data_tree_X),'Actual':data_nontree_y})
x=['ADA','ANN','LOG','SVM']
FINlog = LogisticRegression()
FINlog.fit(result[x],result['Actual'])

total = DataFrame({'ANN':ANN.predict(question_nontree),'SVM':SVM.predict(question_nontree),'GLM':GLM.predict(question_nontree),'ADA':ADA.predict(question_tree)})

submission = pd.DataFrame({'PassengerId': question['PassengerId'], 'Survived':FINlog.predict(total)})
submission.to_csv("C:/Users/Froilan/Desktop/Repository/kaggleResultCSV/Titanic_python_stacking.csv",index=False)

print(data_nontree_X.head())
print(data_tree_X.head())
print(question_nontree)
print(total)
print(submission)