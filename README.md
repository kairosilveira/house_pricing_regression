# House Pricing Regression

This is a repository of a full datascience project focused on MLOps: Deployment, unity tests, OOP, MLFlow. The goal is to predict prices of houses based on the Kings County dataset (avaliable [here]).

 

## analysis/house_pricing.ipynp
In this file you will find a notebook with the steps to find the best model.

## train.py
In this file you will find the code to train the model, including the data preparation and hyperparameter tuning.

## app.py
In this file is the flask code for the API to predict prices, using the model loaded from MLFlow.

## How to run
First, to run the webapp locally you are going to have to clone the repository:

  gh repo clone kairosilveira/house-pricing-regression

Then, create and activate the virtual environment and install the dependencies(make sure eu you have python3 installed):

  python -m venv venv
  source venv/bin/activate #for linux
  venv/Scripts/activate #fow windows
  python -m pip install --upgrade pip
  pip install -r requirements.txt

After installation is done, you can start the flask aplication running the app.py file

    python3 app.py

Now you can use the post method to make predictions using your local server addind /predict to the end point, here is an example using thunder:

![alt text](tests/API_test_thunder.png.png)





[here]: https://www.kaggle.com/harlfoxem/housesalesprediction