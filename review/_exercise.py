import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import (preprocessing, model_selection)

#  Create 100x5 matrix with value from 1->100
np.random.seed(101)
data = np.random.randint(1, 101, (100, 5))
print(data)

# Create 2D vis w/Colorbar and Title
plt.imshow(data, cmap="coolwarm", aspect='auto')
plt.title('title')
plt.colorbar()
plt.show()

# Create pandas dateframe
df = pd.DataFrame(data)
print(df)

# Show scatter plot of col 0 vs col 1
df.plot(x=0, y=1, kind='scatter')
plt.show()

# Scale data to have minimum of 0 and max of 1
scaler = preprocessing.MinMaxScaler()
scaled_data = scaler.fit_transform(data)
print(scaled_data)

# Rename columns, split data into training and test
df.columns = ['f1', 'f2', 'f3', 'f4', 'label']
X = df[['f1', 'f2', 'f3', 'f4']]
y = df[['label']]

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X, y, test_size=.33, random_state=42)

print(X_train)
