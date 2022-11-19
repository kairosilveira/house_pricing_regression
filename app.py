from train import RegressionModel
from flask import Flask,request
import pandas as pd
import mlflow


mlflow.set_experiment("House Pricing Regression")
last_run=dict(mlflow.search_runs().sort_values(by='start_time',ascending=False).iloc[0])
last_model_uri = "mlruns/"+last_run['experiment_id']+"/"+ last_run["run_id"] +'/artifacts/house_price_regression'
model=mlflow.sklearn.load_model(last_model_uri)

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
    response = {"price":list(predictions)}

    return response

if __name__ == "__main__":
    app.run()