import pandas as pd
import numpy as np


def remov_outliers(x_raw,y_raw, whis, cols):
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


def remov_unimportant_cols(df_raw):
    '''Remove unimportant columns for modeling from the dataFrame, if they are present'''
    
    df = df_raw.copy()
    df_cols = list(df.columns)
    unimportant_cols = ["id", "date", "zipcode", "yr_built", "yr_renovated"]
    cols_are_in_df = all(item in df_cols for item in unimportant_cols)

    if not cols_are_in_df:
        return df
    return df.drop(columns=unimportant_cols)


def drop_sqft_above(df_raw):
    '''drop sqft_above to avoid multicollinearity'''
    
    df = df_raw.copy()
    return df.drop(columns = ["sqft_above"])


def create_yr_renovated(df_raw):
    '''create a new feature "yr_renovated"'''

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
    