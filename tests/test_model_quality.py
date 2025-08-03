import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import full_suite # Assuming full_suite is imported

def test_model_quality():
    iris = "./data/iris.csv"
    df = pd.read_csv(iris)

    # Get the name of the label column
    label_name = df.columns[-1]
    X, y = df.drop(columns=[label_name]), df[label_name]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)
    
    # Load model
    model = joblib.load("models/iris_model.pkl")

    # --- FIX STARTS HERE ---

    # 1. Combine features and labels back into single DataFrames for deepchecks
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    # 2. Create Dataset objects correctly
    #    - Pass the combined DataFrame.
    #    - Pass the label column NAME as a string to the `label` parameter.
    dc_train = Dataset(train_df, label=label_name, cat_features=[])
    dc_test = Dataset(test_df, label=label_name, cat_features=[])

    # 3. Run the suite using keyword arguments for both train and test datasets
    suite = full_suite()
    result = suite.run(train_dataset=dc_train, test_dataset=dc_test, model=model)

    # --- FIX ENDS HERE ---

    # Save the report
    result.save_as_html('./tests/deepchecks_report.html',as_widget=False)

    # Fail the test if any checks failed
    # Using is_passed() is a more direct way to check the suite's overall status
    assert result.passed, "Deepchecks suite failed. Check deepchecks_report.html for details."