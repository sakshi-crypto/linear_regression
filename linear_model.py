import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


data = pd.read_csv('ex1data1.txt')
x = data.values[:, 0].reshape(97, 1)
y = data.values[:, 1].reshape(97, 1)
x_train = x[: -20]
x_test = x[-20:]
y_train = y[:-20]
y_test = y[-20:]

regr = LinearRegression()
regr.fit(x_train, y_train)
y_predict = regr.predict(x_test)

# The coefficients
print('Coefficients: ', regr.coef_)
# The mean squared error
print('Mean squared error: '
      % mean_squared_error(y_test, y_predict))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_predict))


plt.scatter(x_test, y_test, color="black")
plt.plot(x_test, y_predict, color="blue")


plt.xticks(())
plt.yticks(())

plt.show()
