
import argparse
import os
import warnings

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelBinarizer, KBinsDiscretizer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.compose import make_column_transformer

from sklearn.exceptions import DataConversionWarning
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings(action="ignore", category=DataConversionWarning)

if __name__ == "__main__":
    #get arguments
    parser = argparse.ArgumentParser()
    args, _ = parser.parse_known_args()
    print("Received arguments {}".format(args))
    
    #get the input data
    input_data_path = os.path.join("/opt/ml/processing/input", "insurance_claims.csv")
    print("Reading input data from {}".format(input_data_path))
    df = pd.read_csv(input_data_path)
    df = pd.DataFrame(data=df)
    print(df.head())

    #replacing ? with nan for the columns
    df['police_report_available']=df['police_report_available'].replace('?',np.nan)
    df['collision_type']=df['collision_type'].replace('?',np.nan)
    df['property_damage']=df['property_damage'].replace('?',np.nan)
    
    #droping rows with nan
    df=df.dropna(subset=['police_report_available', 'collision_type','property_damage'])
    
    #dropping the unnecessary rows
    df=df.drop(['months_as_customer','policy_number','policy_bind_date','policy_csl','auto_year','auto_model','insured_hobbies','insured_zip'],axis=1)
    
    #now deal with the categorical features
    le=LabelEncoder()
    for i in list(df.columns):
        if df[i].dtype=='object':
            df[i]=le.fit_transform(df[i])
    
    #final preprocessed data
    print(df.head())
    train_features_output_path = os.path.join("/opt/ml/processing/output", "preprocessed_data.csv")
    df.to_csv(train_features_output_path, index=False)
    print("done")

    