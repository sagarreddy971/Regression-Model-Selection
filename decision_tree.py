import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_excel("Data_Set_1.xlsx")
x=df.drop(columns=["electrical energy output"])
y=df["electrical energy output"]

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.tree import DecisionTreeRegressor
reg=DecisionTreeRegressor(random_state=42,max_depth=None)
reg.fit(x_train,y_train)
y_pred=reg.predict(x_test)
r2=r2_score(y_test,y_pred)
rmse=np.sqrt(mean_squared_error(y_test,y_pred))
print(f"The r2 score for decision tree regression is: {r2:.2f}")
print(f"The Root mean squared Error is:{rmse:.2f} ")
plt.scatter(y_test,y_pred,color="grey")
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],color="red")
plt.title(f"Decision Tree Regression: Actual vs Predicted\nR2 Score: {r2:.3f}, RMSE: {rmse:.3f}")
plt.show()
print(reg.predict([[15, 44, 1022.06, 78.12]]))