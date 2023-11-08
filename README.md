# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.
2.Upload and read the dataset. 
3.Check for any null values using the isnull() function. 
4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5.Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: S.Sakthi Priya 
RegisterNumber:21222040140  
*/
import pandas as pd
data=pd.read_csv("/content/Employee.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
1.Data head

![s1](https://github.com/SAKTHIPRIYASATHISH/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119104282/538f7dbf-b0d5-4e45-80a8-b45e947390cc)


2.Data set info
![s2](https://github.com/SAKTHIPRIYASATHISH/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119104282/62470c4d-0185-427f-8f33-b016092c13eb)

3.Null dataset

![s3](https://github.com/SAKTHIPRIYASATHISH/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119104282/89315642-f25a-44af-986e-06c6fb8dd39f)


4.Values count from left column

![s4](https://github.com/SAKTHIPRIYASATHISH/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119104282/32cc772a-9540-4f2b-aa40-8e6f935ccf1c)

5.Dataset transformed head

![s5](https://github.com/SAKTHIPRIYASATHISH/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119104282/7ba7306f-c41e-4953-a179-3091d51dcf95)

6.X.head

![s6](https://github.com/SAKTHIPRIYASATHISH/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119104282/b05b4340-3f14-492d-85b8-d7eb8c311773)

7.Accuracy
![s7](https://github.com/SAKTHIPRIYASATHISH/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119104282/d6b54e08-8ed9-49fd-b849-735b739be69f)

8.Data prdiction
![s8](https://github.com/SAKTHIPRIYASATHISH/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119104282/794fc514-e227-459d-ada3-6a7a4d093d61)














## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
