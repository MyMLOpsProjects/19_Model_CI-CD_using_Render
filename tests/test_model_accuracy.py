import joblib
import pandas as pd
from sklearn.datasets import load_iris

def test_model_accuracy():
    iris = "./data/iris.csv"
    df = pd.read_csv(iris)
    X, y = df.iloc[:,:-1],df.iloc[:,-1]    

    model = joblib.load("models/iris_model.pkl")
    score = model.score(X, y)
    
    assert score > 0.9, f"Accuracy too low: {score}"