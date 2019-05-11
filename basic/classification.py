import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

diabetes = pd.read_csv('../__tensorflow__/02-TensorFlow-Basics/pima-indians-diabetes.csv')
# print(diabetes.head())

# Clean data
print(diabetes.columns)
cols_to_norm = ['Number_pregnant', 'Glucose_concentration',
                'Blood_pressure', 'Triceps',
                'Insulin', 'BMI', 'Pedigree']

diabetes[cols_to_norm] = diabetes[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
# print(diabetes.head())

# Continuos Data
number_pregnant = tf.feature_column.numeric_column('Number_pregnant')
glucose_concentration = tf.feature_column.numeric_column('Glucose_concentration')
blood_pressure = tf.feature_column.numeric_column('Blood_pressure')
triceps = tf.feature_column.numeric_column('Triceps')
insulin = tf.feature_column.numeric_column('Insulin')
bmi = tf.feature_column.numeric_column('BMI')
pedigree = tf.feature_column.numeric_column('Pedigree')
age = tf.feature_column.numeric_column('Age')

# Categorical Data
# Easy Grouping -> use "categorical_column_with_vocabulary_list"
assigned_group = tf.feature_column.categorical_column_with_vocabulary_list('Group', ['A', 'B', 'C', 'D'])
# Too many categories -> use "categorical_column_with_hash_bucket"
# assigned_group = tf.feature_column.categorical_column_with_hash_bucket('Group', hash_bucket_size=10)

# View Age data as hist
# diabetes['Age'].hist(bins=20)
# plt.show()

# Convert continuous numeric into buckets
age_bucket = tf.feature_column.bucketized_column(age, boundaries=[20, 30, 40, 50, 60, 70, 80])

feat_cols = [number_pregnant, glucose_concentration, blood_pressure,
             triceps, insulin, bmi, pedigree, assigned_group, age_bucket]

# Train Test split
x_data = diabetes.drop('Class', axis=1)
labels = diabetes['Class']
X_train, X_test, y_train, y_test = train_test_split(x_data, labels, test_size=0.3, random_state=101)

# Train
train_input_fn = tf.estimator.inputs.pandas_input_fn(x=X_train, y=y_train, batch_size=10, num_epochs=1000, shuffle=True)
model = tf.estimator.LinearClassifier(feature_columns=feat_cols, n_classes=2)
model.train(input_fn=train_input_fn, steps=1000)

# Evaluate
eval_input_fn = tf.estimator.inputs.pandas_input_fn(x=X_test, y=y_test, batch_size=10, num_epochs=1, shuffle=False)
results = model.evaluate(input_fn=eval_input_fn)
# print(results)

# Prediction
pred_input_fn = tf.estimator.inputs.pandas_input_fn(x=X_test, batch_size=10, num_epochs=1, shuffle=False)
predictions = model.predict(input_fn=pred_input_fn)
# print(list(predictions))


####### Dense Nearal Network Classifier #######
# When using user defined category column it must be wrapped in an embedding column for NNs since they only operate on numerical values
embedded_group_col = tf.feature_column.embedding_column(assigned_group, dimension=4)
feat_cols = [number_pregnant, glucose_concentration, blood_pressure,
             triceps, insulin, bmi, pedigree, embedded_group_col, age_bucket]

dnn_input_fn = tf.estimator.inputs.pandas_input_fn(X_train, y_train, batch_size=10, num_epochs=1000, shuffle=True)
dnn_model = tf.estimator.DNNClassifier(hidden_units=[10, 10, 10, 10, 10], feature_columns=feat_cols, n_classes=2)
dnn_model.train(input_fn=dnn_input_fn, steps=1000)

dnn_eval_input_fn = tf.estimator.inputs.pandas_input_fn(X_test, y=y_test, batch_size=10, num_epochs=1, shuffle=False)
result = dnn_model.evaluate(dnn_eval_input_fn)
print(result)
