# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Janathul Firdhous A
RegisterNumber:  212224040129
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
Dataset
<img width="97" alt="image" src="https://github.com/user-attachments/assets/4f435be4-9191-4d54-9656-40958cf97f24" />
Head values
<img width="92" alt="image" src="https://github.com/user-attachments/assets/45b14e9a-978a-4c44-b1ee-1834f2da100e" />
Tail values
<img width="86" alt="image" src="https://github.com/user-attachments/assets/906c9ff5-e529-49fe-87ae-52fa8b6bd1d8" />
X and Y values
<img width="41" alt="image" src="https://github.com/user-attachments/assets/a0706a13-c54b-418c-b094-c06d8a7373a5" />
Predication values of X and Y
<img width="353" alt="image" src="https://github.com/user-attachments/assets/29e90383-dc05-41c0-ab9b-3e7f665b4aad" />
MSE,MAE and RMSE
<img width="131" alt="image" src="https://github.com/user-attachments/assets/7cf48655-596e-4748-9c22-8eda5f6afd5a" />
Training set
<img width="422" alt="image" src="https://github.com/user-attachments/assets/b5acdc61-f452-450d-ba46-b291b8e0d175" />
Testing set
<img width="404" alt="image" src="https://github.com/user-attachments/assets/22daf557-6f59-476a-b346-ff0f72de504e" />



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
