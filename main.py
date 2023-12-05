import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("train.csv")

y = df.diabetes.values
x_data = df.drop(['p_id', 'diabetes'], axis=1)
# x_train = df.drop(['p_id', 'diabetes'], axis=1)

# Normalize
x = ((x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))).values

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,stratify=y)


# KNN Model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 2)  # n_neighbors means k
knn.fit(x_train, y_train)
prediction = knn.predict(x_test)
print("knn score: {:.2f}%".format(knn.score(x_test, y_test)*100))

# df = pd.read_csv("test.csv")

# # y_test = df.diabetes.values
# x_data = df.drop(['p_id'], axis=1)
# # x_test = df.drop(['p_id'], axis=1)
# # Normalize
# x_test = ((x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))).values


# df = pd.read_csv("test.csv")
# x_pred = df.drop(['p_id'], axis=1)
# final_prediction = knn.predict(x_pred)

# submission = {'p_id': df.p_id.values, 'diabetes': final_prediction}
# df = pd.DataFrame(submission)

# print(df.head())

# # output to csv for submission
# df.to_csv('submission3.csv', encoding='utf-8', index=False)