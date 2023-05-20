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
![71](https://github.com/Yuvadarshini-Sathiyamoorthy/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/93482485/4128186c-ac76-4c98-ab1e-50329a212049)

## data.info()
![72](https://github.com/Yuvadarshini-Sathiyamoorthy/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/93482485/01908e1f-d893-42e7-a81c-bb1ae1478981)

## isnull() and sum()
![73](https://github.com/Yuvadarshini-Sathiyamoorthy/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/93482485/9f48c14b-79ec-4746-844e-c748175e97ed)

## data.head() for salary
![74](https://github.com/Yuvadarshini-Sathiyamoorthy/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/93482485/fcf0c9c0-a404-4688-ae84-cc722ea2324f)

## MSE value
![75](https://github.com/Yuvadarshini-Sathiyamoorthy/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/93482485/4eb97d21-85d2-41dd-9585-611e6b95836f)

## r2 value
![76](https://github.com/Yuvadarshini-Sathiyamoorthy/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/93482485/88872012-52e0-40af-93f3-3cceb908bb9f)

## data prediction
![77](https://github.com/Yuvadarshini-Sathiyamoorthy/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/93482485/988de6b6-bf23-48f9-a843-7d5cb832cd16)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
