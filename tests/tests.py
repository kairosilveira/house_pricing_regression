from house_pricing_regression.train import RegressionModel
import pytest
import pandas as pd
import os
from unittest.mock import Mock


def test_get_file_path():
    expected = os.path.join(
                        os.path.join(os.path.abspath(""),"data"),
                        "test.csv")  
    result = RegressionModel().get_data_path("test.csv")
    assert result == expected


def test_get_file_path_input_type():
    with pytest.raises(ValueError):
        RegressionModel().get_data_path(35)


@pytest.fixture(scope= "session")
def get_data():
    return RegressionModel().get_data()
    

def test_get_data_cols(get_data):
    expected = pd.Index(['id','date','price','bedrooms','bathrooms','sqft_living','sqft_lot','floors',
    'waterfront','view','condition','grade','sqft_above','sqft_basement','yr_built',
    'yr_renovated','zipcode','lat','long','sqft_living15','sqft_lot15'])
    result = get_data.columns
    assert expected.equals(result)


def test_get_data_type(get_data):
    assert isinstance(get_data, pd.DataFrame)


def test_remov_unimportant_cols():
    df = pd.DataFrame({"col1":[1,2], "col2":[1,2] })
    df2 = RegressionModel().remov_unimportant_cols(df, ["col1"])
    expected = pd.Index(["col2"])
    result = df2.columns
    assert expected.equals(result)

@pytest.fixture(scope= "session")
def get_recommended_data():
    obj = RegressionModel()
    file_path=obj.get_data_path("recommended.csv")
    return obj.get_data(file_path)


def test_remov_recommended(get_recommended_data,get_data):
    df_result = RegressionModel().remov_recommended_houses(get_data)
    ids_recommended = get_recommended_data["id"]
    expected = get_data[~get_data.id.isin(ids_recommended)]["id"]
    result = df_result["id"]
    assert expected.equals(result)


def test_create_yr_no_renovated(get_data):
    cols_expected = list(get_data.columns)
    cols_expected.append("yr_no_renov")
    expected = pd.Index(cols_expected)
    df = RegressionModel().create_yr_no_renovated(get_data)
    result=df.columns
    assert expected.equals(result)




