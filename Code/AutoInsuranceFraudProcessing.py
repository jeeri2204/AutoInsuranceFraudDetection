
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
    #dropping the unnecessary rows
    df=df.dropna(subset=['police_report_available'])
    
    #drop unnecessary columns
    df=df.drop(['months_as_customer'],axis=1)
    
    #now deal with the categorical features
    #for the columns insured_sex and fraud_reported
    le=LabelEncoder()
    for i in list(df.columns):
        if df[i].dtype=='object':
            df[i]=le.fit_transform(df[i])
    
    #final preprocessed data
    print(df.head())
    train_features_output_path = os.path.join("/opt/ml/processing/output", "preprocessed_data.csv")
    df.to_csv(train_features_output_path, index=False)
    print("done")
    
