import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Read in data
census_data = pd.read_csv('../__tensorflow__/02-TensorFlow-Basics/census_data.csv')
# print(census_data.head())


# Transform income_bracket to numerical categories
census_data['income_bracket'] = np.where(census_data['income_bracket'] == ' <=50K', 0, 1)
# print(census_data.head())

# Categorical Data
workclass = tf.feature_column.categorical_column_with_hash_bucket('workclass', 1000)
education = tf.feature_column.categorical_column_with_hash_bucket('education', 1000)
marital_status = tf.feature_column.categorical_column_with_hash_bucket('marital_status', 1000)
occupation = tf.feature_column.categorical_column_with_hash_bucket('occupation', 1000)
relationship = tf.feature_column.categorical_column_with_hash_bucket('relationship', 1000)
race = tf.feature_column.categorical_column_with_hash_bucket('race', 1000)
native_country = tf.feature_column.categorical_column_with_hash_bucket('native_country', 1000)
gender = tf.feature_column.categorical_column_with_vocabulary_list('gender', ['Male', 'Female'])

# Continuous Data
age = tf.feature_column.numeric_column('age')
education_num = tf.feature_column.numeric_column('education_num')
capital_gain = tf.feature_column.numeric_column('capital_gain')
capital_loss = tf.feature_column.numeric_column('capital_loss')
hours_per_week = tf.feature_column.numeric_column('hours_per_week')

# feature Columns
feat_cols = [
    workclass, education, marital_status,
    occupation,  relationship, race, native_country,
    gender, age, education_num, capital_gain,
    capital_loss, hours_per_week
]

# Perform a Train Test Split on the Data
X = census_data.drop('income_bracket', axis=1)
y = census_data['income_bracket']
# print(X.head())
# print(y.head())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Train
train_input_fn = tf.estimator.inputs.pandas_input_fn(x=X_train, y=y_train, batch_size=10, num_epochs=1000, shuffle=True)
model = tf.estimator.LinearClassifier(feature_columns=feat_cols, n_classes=2)
model.train(input_fn=train_input_fn, steps=5000)

# Evaluate
eval_input_fn = tf.estimator.inputs.pandas_input_fn(x=X_test, y=y_test, batch_size=10, num_epochs=1, shuffle=False)
results = model.evaluate(input_fn=eval_input_fn)
# print(results)

# Prediction
pred_input_fn = tf.estimator.inputs.pandas_input_fn(x=X_test, batch_size=10, num_epochs=1, shuffle=False)
predictions = list(model.predict(input_fn=pred_input_fn))
# print(predictions)

final_preds = []
for pred in predictions:
    final_preds.append(pred['class_ids'][0])

print(classification_report(y_test, final_preds))
