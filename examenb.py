from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import pandas as pd





dataset = pd.read_csv('exam_B_dataset.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

lin_reg = LinearRegression()
poly_reg = PolynomialFeatures(degree=4)

X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly,y)
lin_reg.fit(X_poly,y)

plt.scatter(X,y)
plt.plot(X,lin_reg.predict(poly_reg.fit_transform(X)))
plt.show()
