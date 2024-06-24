import streamlit as st
import pandas as pd


def init_state_vars(vars):
    for var in vars:
        if var not in st.session_state:
            st.session_state[var] = None
        

def reset_state(vars):
    for var in vars:
        st.session_state[var] = None
    
@st.cache_data()
def load_csv(file):
    df = pd.read_csv(file)
    st.session_state["df"] = df
    return df


def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')


