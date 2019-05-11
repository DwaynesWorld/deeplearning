import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (mean_squared_error, classification_report)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Read Data
housing_data = pd.read_csv('../__tensorflow__/02-TensorFlow-Basics/cal_housing_clean.csv')
# print(housing_data.head())
print(housing_data.describe().transpose())

# Perform a Train Test Split on the Data
X = housing_data.drop('medianHouseValue', axis=1)
y = housing_data['medianHouseValue']
# print(X.head())
# print(y.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Transform Data
scaler = MinMaxScaler()
scaler.fit(X_train)

X_train = pd.DataFrame(
    data=scaler.transform(X_train),
    columns=X_train.columns,
    index=X_train.index)

X_test = pd.DataFrame(
    data=scaler.transform(X_test),
    columns=X_test.columns,
    index=X_test.index)

# Create Feature Columns
age = tf.feature_column.numeric_column('housingMedianAge')
rooms = tf.feature_column.numeric_column('totalRooms')
bedrooms = tf.feature_column.numeric_column('totalBedrooms')
pop = tf.feature_column.numeric_column('population')
households = tf.feature_column.numeric_column('households')
income = tf.feature_column.numeric_column('medianIncome')

feat_cols = [age, rooms, bedrooms, pop, households, income]

# Train
input_func = tf.estimator.inputs.pandas_input_fn(
    x=X_train,
    y=y_train,
    batch_size=10,
    num_epochs=1000,
    shuffle=True)

model = tf.estimator.DNNRegressor(hidden_units=[6, 10, 6], feature_columns=feat_cols)
model.train(input_fn=input_func, steps=25000)

# Predict
predict_input_func = tf.estimator.inputs.pandas_input_fn(
    x=X_test,
    batch_size=10,
    num_epochs=1,
    shuffle=False)

predictions = list(model.predict(predict_input_func))

# Get Metrics
final_preds = []
for pred in predictions:
    final_preds.append(pred['predictions'])

print(classification_report(y_test, final_preds))
print(mean_squared_error(y_test, final_preds)**0.5)
