from keras.models import Sequential
from keras.layers import *
from pandas_datareader import data, wb
import datetime
from sklearn.model_selection import train_test_split


start = datetime.datetime(2010, 1, 1)
end = datetime.datetime(2012, 7, 15)
yahoo_df = data.DataReader("F", 'yahoo', start, end)
print(yahoo_df.head(5))


training_data_df = yahoo_df.copy()
X = training_data_df.drop(['Volume','Adj Close'], axis=1).values
y = training_data_df[['Adj Close']].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)


# Define the model
model = Sequential()
model.add(Dense(50, input_dim=4, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam')


# Train the model
model.fit(
    X,
    y,
    epochs=50,
    shuffle=True,
    verbose=2
)


test_error_rate = model.evaluate(X_test, y_test, verbose=0)
print("The mean squared error (MSE) for the test data set is: {}".format(test_error_rate))

