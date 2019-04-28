import numpy as np
import pandas as pd
from sklearn import (preprocessing,
                     model_selection)

data = np.random.randint(0, 100, (10, 2))
# print(data)

# Before running data through nn,
# we should be scaling the data

scaler_model = preprocessing.MinMaxScaler()
# print(type(scaler_model))

# scaler_model.fit(data)
# xdata = scaler_model.transform(data)
xdata = scaler_model.fit_transform(data)
# print(xdata)


# Test / Train Split
data = np.random.randint(0, 101, (50, 4))
# print(data)
df = pd.DataFrame(data=data, columns=['f1', 'f2', 'f3', 'label'])
# print(df)

X = df[['f1', 'f2', 'f3']]
y = df[['label']]

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=0.33, random_state=42)

print(X_train.shape)
# print(X_test, y_train, y_test)
