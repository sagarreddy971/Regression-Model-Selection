import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

df=pd.read_excel("Data_Set_1.xlsx")
x=df.drop(columns=["electrical energy output"])
y=df["electrical energy output"]

from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
reg=RandomForestRegressor(n_estimators=100,random_state=42)
reg.fit(x_train,y_train)
y_pred=reg.predict(x_test)
r2=r2_score(y_test,y_pred)
rmse=np.sqrt(mean_squared_error(y_test,y_pred))
print(f"The r2 score for Random Forest Regression is: {r2:.2f}")
print(f"The root mean square error value is: {rmse:.2f}")
plt.scatter(y_test, y_pred, color="darkorange", alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Output")
plt.ylabel("Predicted Output")
plt.title(f"Random forest regression: Actual vs Predicted\nR2 Score: {r2:.3f}, RMSE: {rmse:.3f}")
plt.grid(True)
plt.show()
print(reg.predict([[15, 44, 1022.06, 78.12]]))
print(reg.predict([[14.96,14.96,14.96,14.96]]))
