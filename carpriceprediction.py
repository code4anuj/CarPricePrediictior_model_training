import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

car = pd.read_csv("/content/Cleaned_Car_data.csv")
car.head()

car.shape

car.isnull().sum()

car.describe()

sns.relplot(x='kms_driven',y='Price',data=car)

sns.relplot(x='year',y='Price',data=car)

car.head()

train = car.drop(['name','company','fuel_type'], axis=1)
test= car['Price']

X_train,X_test,y_train,y_test = train_test_split(train,test,test_size=0.3,random_state=2)

regr = LinearRegression()

regr.fit(X_train,y_train)

pred = regr.predict(X_test)

regr.score(X_test,y_test)

plt.scatter(y_test,pred)

