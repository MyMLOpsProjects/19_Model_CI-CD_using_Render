import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np
np.Inf = np.inf # because deepcheck is not able to handle np.Inf, hence np.inf

iris = "./data/iris.csv"
df = pd.read_csv(iris)
X, y = df.iloc[:,:-1],df.iloc[:,-1]

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, './models/iris_model.pkl')
