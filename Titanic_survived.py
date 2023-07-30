"""This is the Linear Regression model to predict the people who survived and who didn't on the bases of 
'Pclass', 'male', 'Age', 'Siblings/Spouses','Parents/Children'and 'Fare' and I used the Sklearn in this model"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
model = LogisticRegression()
pd.options.display.max_columns = 8
df = pd.read_csv(
    "/Users/luqmansabir/Downloads/PYTHON/WebScraping/Scraping a Job Portal/titanic.csv")

df['male'] = df['Sex'] == 'male'
X = df[['Pclass', 'male', 'Age', 'Siblings/Spouses',
        'Parents/Children', 'Fare']].values
y = df['Survived'].values
# fit the model (model buidling)
model.fit(X, y)
# # Make Predicition
print(model.predict(X[:5]))
print(y[:5])
# Scoreing the model (how many predictions are correct)
y_pred = model.predict(X)
print((y == y_pred).sum())
# find the percentage
print(((y == y_pred).sum())/(y.shape[0]))
