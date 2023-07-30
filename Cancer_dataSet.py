from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.datasets import load_breast_cancer
cancer_data = load_breast_cancer()
model = LogisticRegression()
# find out the keys
print(cancer_data.keys())
# create a pandas dataframe with all our feature data
df = pd.DataFrame(cancer_data['data'], columns=cancer_data['feature_names'])
df['target'] = cancer_data['target']

# Build Feature matrix X and target array y
X = df[cancer_data.feature_names].values
y = df['target'].values

# build a model
model.fit(X, y)

# Make a prediction
print("prediction for datapoint 0:", model.predict([X[0]]))

# Scoring the Model
print("Score of the model", model.score(X, y))

# # print the description of data
# print(cancer_data['DESCR'])

# feature data
# print(cancer_data['data'].shape)
