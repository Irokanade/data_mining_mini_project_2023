import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,BaggingClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("train.csv")
y = df.diabetes.values
x = df.drop(['p_id', 'diabetes'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,stratify=y)

rt = RandomForestClassifier(max_depth=5,min_samples_split=3)
rt.fit(x_train,y_train)
rtpredictions = rt.predict(x_test)
print("Random Forest Classifier Score: {:.2f}%".format(rt.score(x_test, y_test)*100))

df = pd.read_csv("test.csv")
x_pred = df.drop(['p_id'], axis=1)
rtpredictions = rt.predict(x_pred)

submission = {'p_id': df.p_id.values, 'diabetes': rtpredictions}
df = pd.DataFrame(submission)

print(df.head())

# output to csv for submission
df.to_csv('submission3.csv', encoding='utf-8', index=False)