from sklearn.linear_model import LogisticRegression
import numpy as np
model = LogisticRegression()
n = int(input())
X = []
for i in range(n):
    X.append([float(x) for x in input().split()])
y = [int(x) for x in input().split()]
datapoint = [float(x) for x in input().split()]

# make the model
model.fit(X, y)

# make the prediciton

Pr = model.predict([datapoint])
print(Pr[0])


# from sklearn.linear_model import LogisticRegression
# import numpy as np
# model = LogisticRegression()
# n = int(input())
# X = []
# for i in range(n):
#     X.append([float(x) for x in input().split()])
# y = [int(x) for x in input().split()]
# datapoint = [float(x) for x in input().split()]

# # make the model
# model.fit(X, y)

# # convert the datapoint into two dimensional array
# datapoint = np.array(datapoint).reshape(1,-1)
# print(model.predict(datapoint[[0]])[0])
