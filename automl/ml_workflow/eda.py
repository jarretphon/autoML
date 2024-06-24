import streamlit as st
from ydata_profiling import ProfileReport

@st.cache_resource()
def generate_report(df):
    pr=ProfileReport(df, explorative=True, dark_mode=True)
    return pr