import inspect
import pickle
import tempfile

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, mean_absolute_error, mean_squared_error, r2_score
from scikitplot.metrics import plot_confusion_matrix, plot_precision_recall, plot_roc
from matplotlib import pyplot as plt
import scipy.stats as stats

CLASSIFICATION_METRICS = [accuracy_score, precision_score, f1_score, recall_score]
REGRESSION_METRICS = [mean_absolute_error, mean_squared_error, r2_score]


def load_data(df, target, train_size):
    y = df[target]
    x = df.drop(columns=[target])
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=train_size, random_state=123)
    return X_train, X_test, y_train, y_test


def setup(df, classification=True):
    with st.container(border=True):
        st.subheader("Set up")
        target = st.selectbox("Select Your target", options=df.columns)
        train_size = st.slider("**Train Test Split**", min_value=0, max_value=100, value=80)
        X_train, X_test, y_train, y_test = load_data(df, target, train_size/100)

        if classification:
            pos_label = st.selectbox("Positive Label", options=df[target].unique())
            return X_train, X_test, y_train, y_test, pos_label
        else:
            return X_train, X_test, y_train, y_test, None


def train_model(ML_algo, X_train, y_train, X_test, classification=True, **kwargs):

    conditional_kwargs = {}

    # Add 'probability' argument for SVC
    if ML_algo == SVC:
        conditional_kwargs["probability"] = True
    
    # Merge the conditional kwargs with the provided kwargs
    all_kwargs = {**kwargs, **conditional_kwargs}

    trained_model = ML_algo(**all_kwargs)
    trained_model.fit(X_train, y_train)
    y_test_pred = trained_model.predict(X_test)
    y_train_pred = trained_model.predict(X_train)

    if classification:
        y_test_prob = trained_model.predict_proba(X_test)
        st.session_state["y_test_prob"] = y_test_prob
        return trained_model, y_test_pred, y_train_pred, y_test_prob
    else:
        return trained_model, y_test_pred, y_train_pred, None


def convert_to_pct(val):
    metric_value = 100 * val
    rounded_val = f"{metric_value}%"
    return rounded_val


def get_val(metrics, y_true, y_pred, pos_label=None):
    
    values = []
    for metric_func in metrics:
        metric_params = inspect.signature(metric_func).parameters
    
        kwargs = {}
        if 'pos_label' in metric_params and pos_label is not None:
            kwargs["pos_label"] = pos_label
        
        metric_value = metric_func(y_true, y_pred, **kwargs)
        values.append(round(metric_value,2))
    
    return values


# show performance metrics of different ML Algo
def show_metric(metrics, y_test, y_test_pred, y_train, y_train_pred, pos_label=None, classification=True):
    test_metric = get_val(metrics, y_test, y_test_pred, pos_label=pos_label)
    train_metric = get_val(metrics, y_train, y_train_pred, pos_label=pos_label)
    
    if classification:
        test_metric = list(map(convert_to_pct, test_metric))
        train_metric = list(map(convert_to_pct, train_metric))
    
    metric_pairs = list(zip(test_metric, train_metric))
    
    cols = st.columns(len(metric_pairs))
    
    for i, metric_pair in enumerate(metric_pairs):
        for j, metric in enumerate(metric_pair):
            with cols[i]:
                id = "Test" if j == 0 else "Train"
                metric_name = (metrics[i].__name__)
                st.metric(f"{id} {metric_name}", value=f"{metric}")


# classification related graphs
def plot_clsf_fig(plot_func, y_test, y_test_val):
    fig = plt.figure(figsize=(6,6))
    axl = fig.add_subplot(111)
    plot_func(y_test, y_test_val, ax=axl)      
    st.pyplot(fig, use_container_width=True)


# regression related graphs
def qq_plot(x):
    fig = plt.figure(figsize=(6, 6))
    stats.probplot(x, dist="norm", plot=plt)
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Sample Quantiles')
    plt.title('Q-Q Plot of Residuals')
    st.pyplot(fig)
    
    
def pred_vs_actual_plot(x, y):
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(x, y, edgecolor='k', alpha=0.7)
    plt.plot([x.min(), x.max()], [x.min(), x.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    st.pyplot(fig)
    

def residual_plot(x,y):
    fig=plt.figure(figsize=(6, 6))
    plt.scatter(x, y, edgecolor='k', alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    st.pyplot(fig)


# Download Trained Model
def download_model(trained_model, placeholder):

    # Save the trained model to a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        pickle.dump(trained_model, tmp)
        tmp_path = tmp.name

    # Read the temporary file content for download button
    with open(tmp_path, "rb") as file:
        model_pkl = file.read()

    placeholder.download_button("Save Model", model_pkl, file_name="trained_model.pkl", mime="application/octet-stream", type="primary", use_container_width=True)


def set_hyperparam_range(hyperparameter):  
    
    hparam = {

        # ___Gradient Boosting hyperparameters___
        "learning_rate": {
            "min_value": 0.01,
            "max_value": 1.0,
            "step": 0.01,
        },
        
        # ____random forest hyperparameters____
        "n_estimators": {
            "min_value": 100,
            "max_value": 1000,
            "step": 1,
        },
        
        "max_depth": {
            "min_value": 1,
            "max_value": 50,
            "step": 1,
        },

        "max_features": {
            "options": ["sqrt", "log2"]
        },
        
        "bootstrap": {
            "options": [True, False]
        },


        # ___SVC hyperparameters___
        "C": {
            "min_value": 1.0,
            "max_value": 50.0,
            "step": 0.1
        },

        "kernel": {
            "options": ["linear", "poly", "rbf", "sigmoid", "precomputed"]
        },

        "degree": {
            "min_value": 0,
            "max_value": 10,
            "step": 1
        },

        "gamma": {
            "options": ["scale", "auto"]
        },


        # __KNN hyperparameters__
        "n_neighbors": {
            "min_value": 1,
            "max_value": 50,
            "step": 1
        },

        "weights": {
            "options": ["uniform", "distance"]
        },

        "metric": {
            "options": ["minkowski"],
            "default": "minkowski",
            "disabled": True
        },

        "p": {
            "min_value": 1.0,
            "step": 0.1
        },

        "algorithm": {
            "options": ['auto', 'ball_tree', 'kd_tree', 'brute']
        }

    }
    
    return hparam[hyperparameter]
    

def get_algo_details(algo_type, algo):

    algorithms = {
        "K-Nearest Neighbours": {
            "model": KNeighborsClassifier if algo_type == "Classification" else KNeighborsRegressor,
            "hyperparameters": ["n_neighbors", "weights", "metric", "p", "algorithm"],
            "performance_metrics": CLASSIFICATION_METRICS if algo_type=="Classification" else REGRESSION_METRICS,
            "classification": True if algo_type == "Classification" else False
        },

        "Random Forest": {
            "model": RandomForestClassifier if algo_type == "Classification" else RandomForestRegressor,
            "hyperparameters": ["n_estimators", "max_depth", "max_features", "bootstrap"],
            "performance_metrics": CLASSIFICATION_METRICS if algo_type=="Classification" else REGRESSION_METRICS,
            "classification": True if algo_type == "Classification" else False
        }, 

        "Support Vector Machine": {
            "model": SVC if algo_type == "Classification" else SVR,
            "hyperparameters": ["C", "kernel", "degree", "gamma"],
            "performance_metrics":  CLASSIFICATION_METRICS if algo_type=="Classification" else REGRESSION_METRICS,
            "classification": True if algo_type == "Classification" else False
        }, 

        "Gradient Boosting": {
            "model": GradientBoostingClassifier if algo_type == "Classification" else GradientBoostingRegressor,
            "hyperparameters": ["n_estimators", "learning_rate", "max_depth"],
            "performance_metrics": CLASSIFICATION_METRICS if algo_type=="Classification" else REGRESSION_METRICS,
            "classification": True if algo_type == "Classification" else False
        },
    }

    return algorithms[algo]


def tune_model(model, hyperparameters, performance_metrics, classification=True):
    
    # Define Page structure
    page_cols = st.columns(2, gap="medium")
    class_rp_placeholder = st.empty()
    curves_placeholder = st.empty()
    save_model_placeholder = st.empty()
    
    # Initialize lists and dictionary
    num_vals = {}
    placeholders = []
    hyperparam_grid = {}
    
    with page_cols[0]:
        
        X_train, X_test, y_train, y_test, pos_label = setup(st.session_state["df"], classification=classification)
        with st.container(border=True):
            st.subheader("Hyperparameters")
            
            #Create number input and hyperparamter value input placeholders for each hyperparameter
            scoring_metric_placeholder = st.empty()
            st.divider()

            for hparam in hyperparameters: 
                
                # Number inputs (users to specify number of hyperparameter values to use) are created for Hyperparameters that accepts a range of values 
                st.write(f"##### {hparam}")
                if hparam in ["n_estimators", "max_depth", "learning_rate", "C", "degree", "n_neighbors", "p"]:
                    num = st.number_input("Number of values", min_value=1, max_value=10, step=1, value=3, key=f"{hparam}")
                    if hparam not in num_vals:
                        num_vals[hparam] = int(num)
                
                # Multiselect menus are created for Hyperparameters that have default options to choose from       
                else:
                    hparam_options = st.multiselect("Choose the parameter", key=f"{hparam}", label_visibility="collapsed", **set_hyperparam_range(hparam))
                    if hparam not in num_vals:
                        num_vals[hparam] =hparam_options
                        
                placeholder = st.empty()
                placeholders.append(placeholder)

                st.divider()
                
            # Create a form for inputting the hyperparameter values
            with st.form("hyperparams", border=False):
                
                scoring_metric = scoring_metric_placeholder.selectbox("Scoring metric", options=["accuracy", "precision", "recall", "f1"] if classification else ["r2", "neg_mean_absolute_error", "neg_mean_squared_error"])
                
                for idx, (hparam, val) in enumerate(num_vals.items()):
                    
                    if isinstance(val, int):
                        # set each hyperparameter's range in number input widget
                        hyperparam_range = set_hyperparam_range(hparam)
                        
                        # Create respective number of columns to hold input fields 
                        cols = placeholders[idx].columns(val)
                        
                        # Create a user-specified number of input fields to take a list of hyperparamter values
                        hparam_values = [   
                            cols[i].number_input(
                                label = f"Value {i+1}", 
                                key=f"{idx}{i}", 
                                **hyperparam_range
                            ) 
                            for i in range(val)
                        ]
                        
                        # populate the dictionary for gridsearch
                        if hparam not in hyperparam_grid:
                            hyperparam_grid[hparam] = hparam_values
                    
                    else:
                        if hparam not in hyperparam_grid:
                            hyperparam_grid[hparam] = val
                
                train = st.form_submit_button("Train Model", use_container_width=True)
                
                if train:
                    
                    # Get the best permutation of hyperparameter
                    gs = GridSearchCV(model(), hyperparam_grid, cv=5, scoring=scoring_metric)
                    gs.fit(X_train, y_train)
                    best_params = gs.best_params_
                    
                    # Train the model with the best permutation of hyperparameters
                    trained_model, y_test_pred, y_train_pred, y_test_prob = train_model(model, X_train, y_train, X_test, classification=classification, **best_params) 
                    
                    # Populate Page with Model metrics and visualisations
                    with page_cols[1]:
                        with st.container(border=True):
                            
                            st.subheader("Best Hyperparameters")
                            st.code(f"{best_params}")
                            
                            st.subheader("Performance Metrics")
                            show_metric(performance_metrics, y_test, y_test_pred, y_train, y_train_pred, pos_label=pos_label, classification=classification)
                            
                            # Plot visualisations specific to classification algos
                            if classification:
                                plot_clsf_fig(plot_confusion_matrix, y_test, y_test_pred)

                                class_rp_placeholder.subheader("Classification Report")
                                class_rp_placeholder.code(f"=={classification_report(y_test, y_test_pred)}")

                                cols = curves_placeholder.columns(2)
                                with cols[0]:  
                                    plot_clsf_fig(plot_roc, y_test, y_test_prob)
                                with cols[1]:  
                                    plot_clsf_fig(plot_precision_recall, y_test, y_test_prob)
                            
                            # Plot visualisations specific to regression algos  
                            else:
                                residuals = y_test - y_test_pred
                                qq_plot(residuals)
                                
                                cols = curves_placeholder.columns(2)
                                with cols[0]:
                                    pred_vs_actual_plot(y_test, y_test_pred)
                                with cols[1]:
                                    residual_plot(y_test_pred, residuals) 

                    download_model(trained_model, save_model_placeholder)