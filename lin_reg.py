import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import train_test_split

df=pd.read_excel("Data_Set_1.xlsx")
x=df.drop(columns=["electrical energy output"])
y=df["electrical energy output"]
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred=reg.predict(x_test)
r2=r2_score(y_test,y_pred)
rmse=np.sqrt(mean_squared_error(y_test,y_pred))
print(f"The r2 score is: {r2:.3f}")
print(f"The root mean square value is: {rmse:.3f}")
plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Output")
plt.ylabel("Predicted Output")
plt.title(f"Linear Regression: Actual vs Predicted\nR2 Score: {r2:.3f}, RMSE: {rmse:.3f}")
plt.grid(True)
plt.show()
print(reg.predict([[15, 44, 1022.06, 78.12]]))
print(reg.predict([[14.96,14.96,14.96,14.96]]))


