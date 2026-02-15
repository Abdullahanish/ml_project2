import  pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import matplotlib.pyplot as plt
import numpy as np

#data loading from csv file 
data=pd.read_csv("students.csv")

#input & output 
X=data[['Study_Hours']]
Y=data['Final_Score']

#object of Linearregression
model=LinearRegression()

#training of data
model.fit(X,Y)
#modle pricdcition for all x values ,so that we can find mae,mse,rmse ,r2
predict=model.predict(X)

mae=mean_absolute_error(Y,predict)
mse=mean_squared_error(Y,predict)
rmse=np.sqrt(mse)
r2=r2_score(Y,predict)

print(" mae : ",mae)
print("mse : ", mse)
print("rmse : ",rmse)
print(" r2 : ",r2)

#graphical representation of data
# plt.figure(figsize=(10,6))
# plt.hist(data['Final_Score'],bins=30,color='green',edgecolor='black')
# plt.title("Final_Score")
# plt.xlabel("along x_axis")
# plt.ylabel("along y axis")
# plt.grid(True)
# plt.show()

#user input
hours=float(input("Enter the hours student study : "))
#model pridiction for one value
result=model.predict([[hours]])
#result of the model means pridected value 
print(result)


