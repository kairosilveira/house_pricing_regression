import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

import mlflow
from mlflow.models.signature import infer_signature

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor



class RegressionModel:

    def __init__(self, mlflow_experiment_name = "House Pricing Regression") -> None:
        self.mlflow_experiment_name = mlflow_experiment_name 
       

    def get_data_path(self, data_file):
        if not isinstance(data_file, str):
            raise ValueError("data_file must be a string")

        data_path = os.path.join(
                        os.path.join(
                            os.path.abspath(""),"data"),
                        data_file)    
        return data_path


    def get_data(self, data_file = "kc_house_data.csv"):
        '''read the data into a dataFrame'''

        data_path = self.get_data_path(data_file)
        df = pd.read_csv(data_path)
        return df


    def remov_unimportant_cols(self, df_raw, unimportant_cols = 
    ["id", "date", "zipcode", "yr_built", "yr_renovated"]):
        '''Remove unimportant columns for modeling from the dataFrame, if they are present'''
        
        df = df_raw.copy()
        df_cols = list(df.columns)
        
        cols_are_in_df = all(item in df_cols for item in unimportant_cols)

        if not cols_are_in_df:
            return df
        return df.drop(columns=unimportant_cols)


    def remov_recommended_houses(self, df_raw):
        '''Remove the instances that were recommended by analysts'''

        df = df_raw.copy()
        data_path = self.get_data_path("recommended.csv")
        df_rec = pd.read_csv(data_path)
        df = df[ ~df.id.isin(df_rec.id) ]
        return df


    def create_yr_no_renovated(self, df_raw): #improve this method later
        '''create a new feature "yr_no_renovated" using "date" yr_build and yr_renovated'''

        df=df_raw.copy()
        df["date"] = pd.to_datetime(df["date"], yearfirst = True)
        auxf = []
        for i in range(len(df["yr_renovated"])):
            if df["yr_renovated"][i]!=0:
                auxf.append(df["yr_renovated"][i])
            else:
                auxf.append(df["yr_built"][i])

        df["auxf"] = auxf
        df["yr_no_renov"] = pd.DatetimeIndex(df["date"]).year - df["auxf"]
        df = df.drop(columns = ["auxf"])

        return df


    def split_data(self, df):
        '''peform train test split where "price" is the target'''

        y = df["price"]
        X  = df.drop(columns=["price"])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        return X_train, X_test, y_train, y_test


    def remov_outliers_iqr(self, x_raw, y_raw, whis, cols=['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 
            'floors', 'sqft_above','sqft_basement', 'lat', 'long', 
            'sqft_living15', 'sqft_lot15','yr_no_renov'] ):
        '''Remove outliers using IQR'''

        df = pd.concat([x_raw, y_raw], axis = 1)
        for col in cols:
            q75 = np.quantile(df[col],0.75)
            q25 = np.quantile(df[col],0.25) 
            iqr = q75-q25
            up_th = whis*iqr+q75
            down_th = q25-whis*iqr
            filt = (df[col]<up_th) & (df[col]>down_th) 
            df = df[filt]
        target = y_raw.name
        return df.drop(columns=[target]), df[target]


    def prepare_data(self, df):
        '''prepare data for training'''

        df = self.create_yr_no_renovated(df)
        df = self.remov_recommended_houses(df)
        df = self.remov_unimportant_cols(df)

        return df


    def get_tuned_model(self, X_train, y_train, model, params):
        '''Tune the hyperparameters and return the best estimator'''

        gs = GridSearchCV(model, params,scoring = 'neg_mean_absolute_error', n_jobs=-1, cv= 4) 
        gs.fit(X_train,y_train)
        return gs.best_estimator_, gs.best_params_


    def get_mae_r2_metrics(self, y, yhat):
        '''get MAE and R2'''

        mae = mean_absolute_error(y, yhat)
        r2 = r2_score(y, yhat)
        metrics = {"MAE":mae, "R2":r2}

        return metrics


    def train(self, X_train, y_train, tuning_hp = True , params = None ):
        '''train the model with the best parameters'''

        if tuning_hp:
            if params == None:
                params = {
                    "n_estimators": [100, 150, 200, 250],
                    "learning_rate":[0.001, 0.01, 0.1],
                    "max_depth":[5, 7, None]
                }

            gbr_model = GradientBoostingRegressor(random_state = 42)
            self.gbr_model, self.params_gbr = self.get_tuned_model(
                X_train, y_train,gbr_model, params)
        elif params==None:
            self.gbr_model = GradientBoostingRegressor(random_state = 42)
            self.gbr_model.fit(X_train, y_train)
            self.params_gbr = self.gbr_model.get_params()
        else:
            self.gbr_model = GradientBoostingRegressor(random_state = 42, **params)
            self.gbr_model.fit(X_train, y_train)   
            self.params_gbr = self.gbr_model.get_params()         

       

    def train_and_save_mlflow_model(self, tuning_hp = True , params = None):
        '''Track the model with MLFlow'''

        mlflow.set_experiment(self.mlflow_experiment_name)
        mlflow.start_run()
        df = self.get_data()
        df = self.prepare_data(df)
        X_train, X_test, y_train, y_test = self.split_data(df)
        self.train(X_train, y_train,tuning_hp, params)
      
        #Log barplot as artifact to save the feature importance plot   
        fi = self.gbr_model.feature_importances_
        plt.figure(figsize=(17,6))
        sns.barplot(x=self.gbr_model.feature_names_in_, y = fi)
        plt.savefig("mlruns/feature_importance.png")
        mlflow.log_artifact("mlruns/feature_importance.png")
        plt.close()

        #logging metrics
        yhat_test = self.gbr_model.predict(X_test)
        self.test_metrics = self.get_mae_r2_metrics(yhat=yhat_test, y=y_test)
        yhat_train = self.gbr_model.predict(X_train)
        self.train_metrics = self.get_mae_r2_metrics(yhat=yhat_train, y=y_train)

        mlflow.log_metric("MSE test",self.test_metrics["MAE"])
        mlflow.log_metric("R2 test",self.test_metrics["R2"])
        mlflow.log_metric("MSE train",self.train_metrics["MAE"])
        mlflow.log_metric("R2 train",self.train_metrics["R2"])

        #logging parameters and model
        params = self.params_gbr
        mlflow.log_params(params)
        signature=infer_signature(X_test,yhat_test)
        mlflow.sklearn.log_model(self.gbr_model, "house_price_regression", signature=signature, )

        mlflow.end_run()
    


if __name__ == "__main__":
    trainer = RegressionModel("House Pricing Regression")
    trainer.train_and_save_mlflow_model()
    print("test metrics:", trainer.test_metrics)
    print("train metrics:",trainer.train_metrics)

