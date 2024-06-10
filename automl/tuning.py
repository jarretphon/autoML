import inspect

import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from matplotlib import pyplot as plt
import scipy.stats as stats


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


def random_forest_hparams(df):
    n_estimators = st.slider("**n estimators**", min_value=100, max_value=1000, value=100, help="The number of trees in the forest.")
    max_depth = st.select_slider("**Max depth**", options=([None]+[i for i in range(51)]), value=None, help="The maximum depth of the tree.")
    max_features = st.selectbox("**Max features**", options=["sqrt", "log2", "None", "Custom"], help="The number of features to consider when looking for the best split: If 'sqrt', then max_features=sqrt(n_features). If 'log2', then max_features=log2(n_features). If None, then max_features=n_features.")
    if max_features == "Custom":
        max_features = st.number_input("**Max Features**", min_value=0, max_value=len(df.columns))
    bootstrap = st.checkbox("**Bootstrap**", help="Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree.")
    
    return n_estimators, max_depth, max_features, bootstrap


def gradient_boosting_hparams():
    n_estimators = st.slider("**n estimators**", min_value=100, max_value=1000, value=100, help="Number of boosting stages.")
    learning_rate = st.slider("**Learning Rate**", min_value=0.01, max_value=1.0, value=0.1, step=0.01, help="Contribution from each tree.")
    max_depth = st.select_slider("**Max depth**", options=([i+1 for i in range(51)]), value=None, help="The maximum depth of the tree.")
    
    return n_estimators, learning_rate, max_depth


def train_model(ML_algo, X_train, y_train, X_test, classification=True, **kwargs):
    trained_model = ML_algo(**kwargs)
    trained_model.fit(X_train, y_train)
    y_test_pred = trained_model.predict(X_test)
    y_train_pred = trained_model.predict(X_train)

    if classification:
        y_test_prob = trained_model.predict_proba(X_test)
        st.session_state["y_test_prob"] = y_test_prob
        return y_test_pred, y_train_pred, y_test_prob
    else:
        return y_test_pred, y_train_pred, None


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
    