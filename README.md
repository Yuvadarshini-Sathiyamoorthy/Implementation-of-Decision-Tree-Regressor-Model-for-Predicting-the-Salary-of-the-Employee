# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.	Import the required packages.
2.	Read the data set.
3.	Apply label encoder to the non-numerical column inoreder to convert into numerical values.
4.	Determine training and test data set.
5.	Apply decision tree regression on to the dataframe and get the values of Mean square error, r2 and data prediction.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Yuvadarshini S
RegisterNumber: 212221230126
*/
```
~~~
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
x.head()

y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
~~~

## Output:
## data.head()
![61](https://github.com/Yuvadarshini-Sathiyamoorthy/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/93482485/34e8ee88-b5bf-4672-b663-76e53c682402)

## data.info()
![62](https://github.com/Yuvadarshini-Sathiyamoorthy/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/93482485/88e4ee22-33f7-45e0-9669-2a79ed715cc2)

## isnull() and sum()
![63](https://github.com/Yuvadarshini-Sathiyamoorthy/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/93482485/23dc4a1f-8d9c-409c-9ba6-be554d58be6b)

## data value counts()
![64](https://github.com/Yuvadarshini-Sathiyamoorthy/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/93482485/8fe71614-cf2f-45d1-9cb4-c8deaf9c8902)

## data.head() for salary
![65](https://github.com/Yuvadarshini-Sathiyamoorthy/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/93482485/ac4e6c9d-2153-4556-a788-81466539e51f)

## x.head()
![66](https://github.com/Yuvadarshini-Sathiyamoorthy/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/93482485/09f0c774-63ba-417f-9eff-4624ecb62e3a)

## accuracy value
![67](https://github.com/Yuvadarshini-Sathiyamoorthy/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/93482485/44ac773e-10b1-4819-be20-84b4e3b4738a)

## data prediction
![68](https://github.com/Yuvadarshini-Sathiyamoorthy/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/93482485/977178ae-d688-40f3-9bfe-64519c05ce11)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
