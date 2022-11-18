from train import RegressionModel
from flask import Flask,request
import mlflow
import pandas as pd


mlflow.set_experiment("House Pricing Regression")
last_run=dict(mlflow.search_runs().sort_values(by='start_time',ascending=False).iloc[0])
artifact_uri=last_run['artifact_uri']
model=mlflow.sklearn.load_model(artifact_uri+'/house_price_regression')

app = Flask(__name__)

@app.route("/predict", methods = ["POST"])
def predict():
    data_json = request.get_json()
    data = data_json["data"]
    df = pd.DataFrame(data)

    preparator = RegressionModel()
    df = preparator.create_yr_no_renovated(df)
    df = preparator.remov_unimportant_cols(df,["yr_built", "date", "yr_renovated"])
    predictions=model.predict(df)

    return list(predictions)

if __name__ == "__main__":
    app.run()