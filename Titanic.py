import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn import svm

#load training data
train_data=pd.read_csv("/home/maheep/Videos/titanic/train.csv")
#drop few features 
train_data = train_data.drop(["PassengerId","Name","Ticket","Cabin"],axis=1)

#find mean and mode of and replace NAN by mean value or mode value
male = train_data.loc[(train_data['Sex'] >= "male")]
male_mean = male['Age'].mean()
female = train_data.loc[(train_data['Sex'] >= "female")]
female_mean = female['Age'].mean()
train_data['Age'] = np.where( (train_data['Age'].isnull()) & (train_data['Sex'] == 'male'), male_mean, train_data['Age'])
train_data['Age'] = np.where( (train_data['Age'].isnull()) & (train_data['Sex'] == 'female'), female_mean, train_data['Age'])


#x_Age_mode = train_data['Age'].mean()
#train_data['Age'] = np.where(train_data['Age'].isnull(), x_Age_mode , train_data['Age'])


x_embarked_mode = train_data['Embarked'].mode()
train_data['Embarked'] = np.where(train_data['Embarked'].isnull(), x_embarked_mode , train_data['Embarked'])

#create a LabelEncoder object and fit it to each feature which contain textual data

number=LabelEncoder()
train_data['Sex']=number.fit_transform(train_data["Sex"].astype('str'))
train_data['Embarked']=number.fit_transform(train_data["Embarked"].astype('str'))

#convert Age and Fare into integers

train_data['Age']=train_data["Age"].astype(dtype=np.int64)
train_data['Fare']=train_data["Fare"].astype(dtype=np.int64)

#Divide all features from class
array = train_data.values
X_train = array[:,1:8]
Y_train = array[:,0]

#create a OneHotEncoder object, and fit it to all of training data

# 1. INSTANTIATE
enc = preprocessing.OneHotEncoder()

# 2. FIT
enc.fit(X_train)

# 3. Transform
onehotlabels_train = enc.transform(X_train).toarray()
onehotlabels_train.shape

#load test data

test_data=pd.read_csv("/home/maheep/Videos/titanic/test.csv")
#drop few features 
test_data = test_data.drop(["PassengerId","Name","Ticket","Cabin"],axis=1)

#find mean and mode of and replace NAN by mean value or mode value
male = test_data.loc[(test_data['Sex'] >= "male")]
male_mean = male['Age'].mean()
female = test_data.loc[(test_data['Sex'] >= "female")]
female_mean = female['Age'].mean()
test_data['Age'] = np.where( (test_data['Age'].isnull()) & (test_data['Sex'] == 'male'), male_mean, test_data['Age'])
test_data['Age'] = np.where( (test_data['Age'].isnull()) & (test_data['Sex'] == 'female'), female_mean, test_data['Age'])


#x_Age_mode = test_data['Age'].mean()
#test_data['Age'] = np.where(test_data['Age'].isnull(), x_Age_mode , test_data['Age'])

x_fare_mode = test_data['Fare'].mean()
test_data['Fare'] = np.where(test_data['Fare'].isnull(), x_fare_mode , test_data['Fare'])

x_embarked_mode = test_data['Embarked'].mode()
test_data['Embarked'] = np.where(test_data['Embarked'].isnull(), x_embarked_mode , test_data['Embarked'])

#create a LabelEncoder object and fit it to each feature which contain textual data

number=LabelEncoder()
test_data['Sex']=number.fit_transform(test_data["Sex"].astype('str'))
test_data['Embarked']=number.fit_transform(test_data["Embarked"].astype('str'))

#convert Age and Fare into integers

test_data['Age']=test_data["Age"].astype(dtype=np.int64)
test_data['Fare']=test_data["Fare"].astype(dtype=np.int64)

#Divide all feature from class
array = test_data.values
X_test = array[:,0:7]

#create a OneHotEncoder object, and fit it to all of testing data

# 1. INSTANTIATE
enc = preprocessing.OneHotEncoder()

# 2. FIT
enc.fit(X_test)

# 3. Transform
onehotlabels_test = enc.transform(X_test).toarray()
onehotlabels_test.shape

# Create SVM classification object 
model = svm.SVC(kernel='linear', gamma=1) 
#fit training data into model
model.fit(X_train, Y_train)

#Predict Output
predicted= model.predict(X_test)


#For creating submission file
test_data=pd.read_csv("/home/maheep/Videos/titanic/test.csv")

#create submission file
submission = np.empty((418,2),dtype=int)
submission[:,0] = test_data["PassengerId"]
submission[:,1] = predicted
submission = pd.DataFrame(data=submission,columns=["PassengerId","Survived"])
submission.to_csv("submission.csv",index = False)
