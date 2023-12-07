#Computer Science Capstone Project
#install pandas using terminal
#instal matplotlib using terminal
#Add Scikit-learn library
import numpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#load data into dataframe
df = pd.read_csv('house.csv')

#split the data

y = df.price
x = df.drop(columns= ['price'])

#normalize the data using scalar
from sklearn.preprocessing import StandardScaler

#scaler = StandardScaler()
#x_scaled = scaler.fit_transform(x)

#train and test split
x_train, x_test, y_train, y_test = train_test_split(x.values, y, test_size=0.3, random_state=666)

#build the multiple linear regression algorithm

MLR = LinearRegression()
MLR.fit(x_train, y_train)

#predict prices using test set
y_pred = MLR.predict(x_test)
print(y_pred)
#shows square meters versus sales price in table.
plt.figure()
plt.title("Square meters versus sales price")
x = df["net_sqm"]
y = df["price"]
plt.plot(x,y)
plt.show()

#shows bedroom count versus price
plt.figure()
plt.title("Number of rooms versus sales price")
x = df["bedroom_count"]
y = df["price"]
plt.plot(x,y)
plt.show()

#show predicted versus test data
plt.figure()
plt.plot(y_test, y_pred)
plt.title("Test values compared to values predicted by the model")
plt.show()

#show head of data being used
print("The type of data used for the model:")
print(df.head())

#print accuracy of predictions versus known data
print("Regression models R^2 score:  " + str(MLR.score(x_test, y_pred)))

stop = False
while stop == False:
    bedroom_count = input("How many bedrooms does the property have?")
    footage = input("What is the net square meters?")
    center_distance = input("How far is it from the city center?")
    floor = input("What floor is the property on?")
    age = input("How old is the property")

    #new_property = scaler.fit_transform(np.array([[0, bedroom_count, footage, center_distance, floor, age]], dtype=float))
    new_property = np.array([[0, bedroom_count, footage, center_distance, floor, age]], dtype=float)
    print(MLR.predict(new_property))

    quit = input("Would you like to examine another property? Y/N")
    if quit == "N":
        stop = True