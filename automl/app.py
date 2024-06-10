import streamlit as st 
from streamlit_option_menu import option_menu
from streamlit_pandas_profiling import st_profile_report
import pandas as pd
from ydata_profiling import ProfileReport

from pycaret.regression import RegressionExperiment, setup, compare_models, pull
from pycaret.classification import ClassificationExperiment, setup, compare_models, pull
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, classification_report, precision_score, f1_score, recall_score, mean_absolute_error, mean_squared_error, r2_score
import scikitplot as skplt
from scikitplot.metrics import plot_confusion_matrix, plot_precision_recall, plot_roc
from matplotlib import pyplot as plt
import scipy.stats as stats

from data_cleaning import categorise_df, convert_pct, rm_cols, handle_impute, remove_duplicates, handle_outliers, encode_by, transform_by, scale_by, aggregate_by
from compare import compare_algo
from tuning import gradient_boosting_hparams, pred_vs_actual_plot, qq_plot, residual_plot, show_metric, train_model, setup, random_forest_hparams, plot_clsf_fig
from util import load_csv, reset_state, init_state_vars, convert_df

@st.cache_resource()
def generate_report(df):
    pr=ProfileReport(df, explorative=True, dark_mode=True)
    return pr

VARS = ["df", "imputed", "duplicates", "outliers", "encoder", "scaler", "transformer", "aggregator", "y_test_pred", "y_test_prob"]
CLASSIFICATION_METRICS = [accuracy_score, precision_score, f1_score, recall_score]
REGRESSION_METRICS = [mean_absolute_error, mean_squared_error, r2_score]

st.set_page_config("Auto ML", layout="wide")

init_state_vars(VARS)

st.markdown("""
    <style>
        .st-emotion-cache-ocqkz7.e1f1d6gn5{
            align-items: center
        }
        
        .st-emotion-cache-1wivap2.e1i5pmia3{
            font-size: 1.75rem;
        }
    </style>    
""", unsafe_allow_html=True)

st.title("Auto Machine Learning")
st.divider()

sidebar = st.sidebar
with sidebar:
    nav = option_menu("Auto ML", options=["Data Exploration", "Data Cleaning", "Compare Models", "Tuning and Optimisation"], menu_icon=["robot"], icons=["clipboard-data", "binoculars", "diagram-3", "tools"])
    file = st.file_uploader("Dataset", type=["csv", "xlsx"], on_change=reset_state, args=(VARS,))


if file is not None: 
    df=load_csv(file)
    
    numeric_df, categorical_df = categorise_df(st.session_state["df"])

    if nav == "Data Exploration":
        st.header("Exploratory Data Analysis")

        with st.container(border=True):
            with st.expander("See DataFrame"):
                st.dataframe(st.session_state["df"], use_container_width=True)

            st.subheader("Data Visualiser")
            #if st.button("Generate Report"):
            #pr=ProfileReport(st.session_state["df"], explorative=True, dark_mode=True)
            pr = generate_report(st.session_state["df"])
            st_profile_report(pr)


    elif nav == "Data Cleaning":
        st.header("Data Cleaning and Preprocessing")

        with st.container(border=True):

            #Filter necessary columns
            strat_col, action_col = st.columns([3,2])
            with strat_col:
                with st.container(border=True):
                    useless_cols = st.multiselect("Remove Unnecessary Columns", options=st.session_state["df"].columns)
            with action_col:
                st.button("Remove Columns", on_click=rm_cols, args=(st.session_state["df"], useless_cols))
                    
            # Convert Currency and percentage
            strat_col, action_col = st.columns([3,2])
            with strat_col:  
                with st.container(border=True):
                    col_selected = st.multiselect("Select Currency/Percentage Column", options=st.session_state["df"].columns)
                    conversion_strategy = st.radio("Convert Currency and Percentages", options=["Remove $ and convert to float", "Convert % to float"], horizontal=True, help="Remove duplicate rows in your dataset.")
            with action_col:
                st.button("Convert to float", on_click=convert_pct, args=(conversion_strategy, st.session_state["df"], col_selected))


            # Handle Missing Values
            strat_col, action_col = st.columns([3,2])
            with strat_col:
                with st.container(border=True):
                    cat_imputation_options = ["most frequent", "drop rows"]
                    num_imputation_options = ["mean", "median", "most frequent", "drop rows"]

                    cat_cols_with_na = categorical_df.columns[categorical_df.isna().any()].tolist()
                    num_cols_with_na = numeric_df.columns[numeric_df.isna().any()].tolist()

                    cat_select_placeholder = st.empty()
                    cat_imp_placeholder = st.empty()
                    num_select_placeholder = st.empty()
                    num_imp_placeholder = st.empty()

                    selected_cat_col = cat_select_placeholder.multiselect("Select categorical columns to handle", options=cat_cols_with_na)
                    cat_imputation_stategy = cat_imp_placeholder.radio("Imputation Strategy (Categorical)", options=cat_imputation_options, horizontal=True, help="This represents how missing values are being handled in the dataset.", key=1)
                    
                    selected_num_col = num_select_placeholder.multiselect("Select numeric columns to handle", options=num_cols_with_na)
                    num_imputation_strategy = num_imp_placeholder.radio("Imputation Strategy (Numerical)", options=num_imputation_options, horizontal=True, help="This represents how missing values are being handled in the dataset.", key=2)

                    if cat_imputation_stategy == "drop rows":
                        cat_select_placeholder.empty()
                        num_select_placeholder.empty()
                        num_imputation_strategy = num_imp_placeholder.radio("Imputation Strategy (Numerical)", options=["drop rows"], horizontal=True, help="This represents how missing values are being handled in the dataset.", key=3)
                        
                    elif num_imputation_strategy == "drop rows":  
                        cat_select_placeholder.empty()
                        num_select_placeholder.empty()
                        cat_imputation_stategy = cat_imp_placeholder.radio("Imputation Strategy (Categorical)", options=["drop rows"], horizontal=True, help="This represents how missing values are being handled in the dataset.", key=4)
            
            with action_col:
                st.button("Impute", on_click=handle_impute, args=(st.session_state["df"], cat_imputation_stategy, selected_cat_col, num_imputation_strategy, selected_num_col))
                
            if isinstance(st.session_state["imputed"], pd.DataFrame):
                st.info(f"No. of rows dropped: {st.session_state['imputed'].shape[0]}")
                with st.expander("See Dropped Rows"):
                    st.dataframe(st.session_state["imputed"], use_container_width=True)
            elif isinstance(st.session_state["imputed"], int):
                st.info(f"No. of values imputed: {st.session_state['imputed']}")
                    

            # Remove Duplicates
            strat_col, action_col = st.columns([3,2])
            with strat_col:  
                with st.container(border=True):
                    duplicates_strategy = st.radio("Remove Duplicates", options=["Yes", "No"], horizontal=True, help="Remove duplicate rows in your dataset.")
            with action_col:
                st.button("Remove Duplicates", on_click=remove_duplicates, args=(st.session_state["df"], "Yes"))

            if isinstance(st.session_state["duplicates"], pd.DataFrame):
                st.info(f"No. of duplicates removed: {st.session_state['duplicates'].shape[0]}")
                with st.expander("See removed duplicates"):
                    st.dataframe(st.session_state["duplicates"], use_container_width=True)        


            # Handle Outliers
            strat_col, action_col = st.columns([3,2])
            with strat_col: 
                with st.container(border=True):
                    outlier_strategy = st.radio("Handle Outliers", options=["Yes", "No"], horizontal=True, help="Remove outliers in your dataset using Z-score.")
                    if outlier_strategy == "Yes":
                        threshold = st.slider("Threshold (σ)", min_value=1, max_value=5, value=3)
            with action_col:
                st.button("Remove Outliers", on_click=handle_outliers, args=(st.session_state["df"], "Yes", threshold))
                    
            if isinstance(st.session_state["outliers"], pd.DataFrame):
                st.info(f"Number of outliers removed: {st.session_state['outliers'].shape[0]}")
                with st.expander("See removed outliers"):
                    st.dataframe(st.session_state["outliers"], use_container_width=True)   


            #Encode Categorical Variables
            strat_col, action_col = st.columns([3,2])
            with strat_col: 
                with st.container(border=True):
                    cat_col = st.multiselect("Select categorical columns to encode", options=categorical_df.columns)
                    encode_strategy = st.radio("Encode Categorical Variables", options=["One Hot Encoding", "Label Encoding"], horizontal=True)
            with action_col:
                st.button("Encode", on_click=encode_by, args=(encode_strategy, st.session_state["df"], cat_col))

            if st.session_state["encoder"]:
                st.success(f"{st.session_state['encoder']} applied successfully")


            #Custom Data Transformation
            strat_col, action_col = st.columns([3,2])
            with strat_col: 
                with st.container(border=True):
                    num_col = st.multiselect("Select numerical columns to transform", options=numeric_df.columns)
                    transformation_strategy = st.radio("Transformation options", options=["Log Transformation", "Square Transformation", "Square Root Transformation"], horizontal=True)
            with action_col:
                st.button("Transform", on_click=transform_by, args=(transformation_strategy, st.session_state["df"], num_col)) 

            if st.session_state["transformer"]:
                st.success(f"{st.session_state['transformer']} applied successsfully")

            #Scaling
            strat_col, action_col = st.columns([3,2])
            with strat_col: 
                with st.container(border=True):
                    scaling_strategy = st.radio("Scaling options", options=["Normalisation", "Standardisation"], horizontal=True)
            with action_col:
                st.button("Scale", on_click=scale_by, args=(scaling_strategy, st.session_state["df"])) 

            if st.session_state["scaler"]:
                st.success(f"{st.session_state['scaler']} applied successfully")


            #Custom Feature Engineering
            strat_col, action_col = st.columns([3,2])
            with strat_col: 
                with st.container(border=True):
                    agg_cols = st.multiselect("Select columns to aggregate", options=st.session_state["df"].columns)
                    aggregation_strategy = st.radio("Aggregation options", options=["➕", "➖", "✖️", "➗"], horizontal=True)
                    new_feature_name = st.text_input("Name of new aggregated feature", placeholder="Feature 1")
            with action_col:
                st.button("Create", on_click=aggregate_by, args=(aggregation_strategy, st.session_state["df"], agg_cols, new_feature_name))
           
            if st.session_state["aggregator"]:
                st.success(f"{st.session_state['aggregator'][0]} applied. {st.session_state['aggregator'][1]} created successfully")
        

        with st.container(border=True):
            st.subheader("Cleaned Data")
            st.dataframe(st.session_state["df"], use_container_width=True)
            csv = convert_df(st.session_state["df"])
            file_name = f"{file.name.split('.')[0]}_cleaned.csv"
            st.download_button("Download as CSV", data=csv, file_name=file_name, mime="text/csv")


    elif nav == "Compare Models":
        st.header("Compare Models")
        target = st.selectbox("Choose your target", options=st.session_state["df"].columns)
        
        if st.button("Train models"):
            
            if pd.api.types.is_numeric_dtype(st.session_state["df"][target]):
                exp = RegressionExperiment()       
            else:
                exp=ClassificationExperiment()
                
            best_model = compare_algo(st.session_state["df"], target, exp)
            st.dataframe(best_model, use_container_width=True)
       
            
    elif nav == "Tuning and Optimisation":
        
        algo = sidebar.selectbox("Select Algorithm", options=["Random Forest Classifier", "Gradient Boosting Regressor"])
            
        st.header("Tuning and Optimisation")
       
        with st.container(border=True):
            cols = st.columns(2, gap="medium")
            class_rp = st.empty()
            curves = st.empty()
            
            if algo == "Random Forest Classifier":
                with cols[0]:
                    with st.form("Hyperparameter Tuning", border=False):
                        X_train, X_test, y_train, y_test, pos_label = setup(st.session_state["df"])
                        with st.container(border=True):
                            st.subheader("Hyperparameters")
                            n_estimators, max_depth, max_features, bootstrap = random_forest_hparams(st.session_state["df"])
                            train = st.form_submit_button("Train Model", type="primary", use_container_width=True)
                        
                    if train:
                        y_test_pred, y_train_pred, y_test_prob = train_model(RandomForestClassifier, X_train, y_train, X_test, n_estimators=n_estimators, max_depth=max_depth, max_features=max_features, bootstrap=bootstrap) 
                        
                        with cols[1]:
                            with st.container(border=True):
                                st.subheader("Performance Metrics")
                                show_metric(CLASSIFICATION_METRICS, y_test, y_test_pred, y_train, y_train_pred, pos_label=pos_label)
                                plot_clsf_fig(plot_confusion_matrix, y_test, y_test_pred)

                        class_rp.subheader("Classification Report")
                        class_rp.code(f"=={classification_report(y_test, y_test_pred)}")
                
                        cols = curves.columns(2)
                        with cols[0]:  
                            plot_clsf_fig(plot_roc, y_test, y_test_prob)
                        with cols[1]:  
                            plot_clsf_fig(plot_precision_recall, y_test, y_test_prob)
            

            elif algo == "Gradient Boosting Regressor":
                with cols[0]:
                    with st.form("Hyperparameter Tuning", border=False):
                        X_train, X_test, y_train, y_test, _ = setup(st.session_state["df"], classification=False)
                        with st.container(border=True):
                            st.subheader("Hyperparameters")
                            n_estimators, learning_rate, max_depth = gradient_boosting_hparams()
                            train = st.form_submit_button("Train Model", type="primary", use_container_width=True)

                    if train:
                        y_test_pred, y_train_pred, _ = train_model(GradientBoostingRegressor, X_train, y_train, X_test, classification=False, n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth) 
                        
                        with cols[1]:
                            with st.container(border=True):
                                st.subheader("Performance Metrics")
                                show_metric(REGRESSION_METRICS, y_test, y_test_pred, y_train, y_train_pred, classification=False)
                                residuals = y_test - y_test_pred
                                qq_plot(residuals)
                                
                            cols = curves.columns(2)
                            with cols[0]:  
                                pred_vs_actual_plot(y_test, y_test_pred)
                            with cols[1]:  
                                residual_plot(y_test_pred, residuals)

else:           
    st.info("Upload a Dataset")