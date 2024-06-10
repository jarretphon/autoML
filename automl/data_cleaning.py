import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler, FunctionTransformer
from scipy import stats

def categorise_df(df):
    numeric_df = df.select_dtypes(include=np.number)
    categorical_df = df.select_dtypes(exclude=np.number)

    return numeric_df, categorical_df


def rm_cols(df, selected_cols):
    df.drop(columns=selected_cols, inplace=True)
    st.session_state["df"] = df


def convert_pct(strategy, df, cols):

    symbols = r'[\$%]'
    for col in cols:
        df[col] = df[col].str.replace(symbols, '', regex=True)
        df[col] = pd.to_numeric(df[col])

        if strategy == "Convert % to float":
            df[col] = df[col]/100
   
    st.session_state["df"] = df


def impute_by(imputation_strat, df, col=None):

    if imputation_strat not in ["most frequent", "mean", "median", "drop rows"]:
        return

    if imputation_strat == "drop rows":
        empty_rows = df[df.isna().any(axis=1)]
        df.dropna(inplace=True)
        df.reset_index(inplace=True, drop=True)
        return empty_rows

    elif imputation_strat in ["mean", "median", "most frequent"]:
        if imputation_strat == "most frequent":
            imputation_strat = "most_frequent"
        imputer = SimpleImputer(missing_values=np.nan, strategy=imputation_strat)
        imputed_column = imputer.fit_transform(df[col])
        df[col] = imputed_column
        return df


def handle_impute(df, cat_impute_strat, cat_col, num_impute_strat, num_col):
    
    if cat_impute_strat == "drop rows":
        empty_rows = impute_by(cat_impute_strat, df)
        st.session_state["df"] = df
        st.session_state["imputed"] = empty_rows
       
    else:
        numerical_df, categorical_df = categorise_df(df)
        try: 
            num_df_imputed = impute_by(num_impute_strat, numerical_df, num_col)
        except ValueError:
            num_df_imputed = numerical_df

        try: 
            cat_df_imputed = impute_by(cat_impute_strat, categorical_df, cat_col)
        except ValueError:
            cat_df_imputed = categorical_df

        df_imputed = pd.concat([num_df_imputed, cat_df_imputed], axis=1)
        values_imputed = df.isna().sum().sum() - df_imputed.isna().sum().sum()
        st.session_state["df"] = df_imputed
        st.session_state["imputed"] = int(values_imputed)


def remove_duplicates(df, strategy):
    if strategy == "Yes":
        duplicates = df[df.duplicated(keep="first")]
        df.drop_duplicates(inplace=True)
        df.reset_index(inplace=True, drop=True)
        st.session_state["df"] = df
        st.session_state["duplicates"] = duplicates
        return duplicates


def handle_outliers(df, strategy, threshold=3):
    if strategy == "Yes":
        numeric_df = df.select_dtypes(include=np.number)

        z_scores = stats.zscore(numeric_df)
        abs_z_scores = np.abs(z_scores)
        outlier_indices = np.where(abs_z_scores > threshold)
        
        outlier_row_index = list(set(outlier_indices[0]))
        outlier_rows = df.loc[outlier_row_index]
        df.drop(df.index[outlier_row_index], inplace=True)
        df.reset_index(inplace=True, drop=True)

        st.session_state["df"] = df
        st.session_state["outliers"] = outlier_rows
        return outlier_rows


def encode_by(strategy, df, cols):
    if strategy == "One Hot Encoding":
        enc = OneHotEncoder(sparse_output=False)
        encoded_data = enc.fit_transform(df[cols])
        encoded_df = pd.DataFrame(encoded_data, columns=enc.get_feature_names_out(cols))
        encoded_df = pd.concat([df, encoded_df], axis=1)
        encoded_df.drop(columns=cols, inplace=True)
        st.session_state["df"] = encoded_df
        st.session_state["encoder"] = strategy

    elif strategy == "Label Encoding":
        enc = LabelEncoder()
        for col in cols:
            encoded_data = enc.fit_transform(df[col])
            #df[col] = pd.DataFrame(encoded_data)
            df[col] = encoded_data
        st.session_state["df"] = df    
        st.session_state["encoder"] = strategy   


def transform_by(strategy, df, cols):
    
    if strategy == "Log Transformation":
        func = np.log1p
        
    elif strategy == "Square Transformation":
        func=np.square
        
    elif strategy == "Square Root Transformation":
        func=np.sqrt
        
    transformer = FunctionTransformer(func=func)
    df[cols] = pd.DataFrame(transformer.fit_transform(df[cols]))
    st.session_state["df"] = df
    st.session_state["transformer"] = strategy


def scale_by(strategy, df):
    numeric_df, categorical_df = categorise_df(df)

    if strategy == "Normalisation":
        scaler = MinMaxScaler()
    elif strategy == "Standardisation":
        scaler = StandardScaler()
        
    scaled_data = scaler.fit_transform(numeric_df)
    scaled_num_df = pd.DataFrame(scaled_data, columns=numeric_df.columns)
    scaled_df = pd.concat([scaled_num_df, categorical_df], axis=1)
    st.session_state['df'] = scaled_df
    st.session_state["scaler"] = strategy


def aggregate_by(strategy, df, cols, new_col_name):

    if strategy == "➕":
        df[new_col_name] = df[cols].sum(axis=1)
    elif strategy == "➖":
        df[new_col_name] = df[cols].apply(lambda row: np.abs(row[0] - row[1:].sum()), axis=1)
    elif strategy == "✖️":
        df[new_col_name] = df[cols].prod(axis=1)
    else:
        df[new_col_name] = df[cols].apply(lambda row: row[0] / row[1:].prod(), axis=1)
    
    st.session_state["df"] = df
    st.session_state["aggregator"] = (strategy, new_col_name)
    

