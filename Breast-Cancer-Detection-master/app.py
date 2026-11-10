
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)


# ==========================================================
# 1. PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Breast Cancer Detection | AI Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# 2. CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

.stApp {
    background-color: #081827;
    color: #EAF2F8;
}

[data-testid="stSidebar"] {
    background-color: #102638;
}

h1, h2, h3 {
    color: #EAF2F8 !important;
}

p, label {
    color: #C4D5E0;
}

div[data-testid="stMetric"] {
    background-color: #142F43;
    border: 1px solid #28516A;
    padding: 18px;
    border-radius: 12px;
}

div[data-testid="stMetricValue"] {
    color: #42D6C5;
}

.stButton button {
    background-color: #159E94;
    color: white;
    border-radius: 8px;
    border: none;
    font-weight: bold;
}

.stButton button:hover {
    background-color: #20B8A9;
    color: white;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #28516A;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================================
# 3. LOAD DATASET
# ==========================================================

@st.cache_data
def load_data():

    file_path = "data.csv"

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            "data.csv not found. Keep data.csv in the same "
            "directory as app.py."
        )

    data = pd.read_csv(file_path)

    # Remove unnecessary columns
    data = data.drop(
        columns=["id", "Unnamed: 32"],
        errors="ignore"
    )

    # Remove completely empty columns
    data = data.dropna(axis=1, how="all")

    if "diagnosis" not in data.columns:
        raise ValueError("The dataset must contain diagnosis column.")

    data["diagnosis"] = (
        data["diagnosis"].astype(str).str.strip().str.upper()
    )

    data = data[data["diagnosis"].isin(["B", "M"])].copy()

    # Convert all feature columns to numeric
    feature_columns = [
        col for col in data.columns if col != "diagnosis"
    ]

    for col in feature_columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(
        subset=feature_columns + ["diagnosis"]
    ).reset_index(drop=True)

    if data.empty:
        raise ValueError("No valid dataset records were found.")

    return data


# ==========================================================
# 4. TRAIN SVM MODEL
# ==========================================================

@st.cache_resource
def train_model(data):

    X = data.drop(columns=["diagnosis"])
    y = data["diagnosis"]

    # Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Pipeline prevents preprocessing leakage
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(
            kernel="rbf",
            probability=True,
            random_state=42
        ))
    ])

    # Hyperparameter optimization
    param_grid = {
        "svc__C": [0.1, 1, 10],
        "svc__gamma": ["scale", 0.01, 0.1, 1]
    }

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        refit=True
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    predictions = best_model.predict(X_test)

    probabilities = best_model.predict_proba(X_test)

    malignant_index = list(best_model.classes_).index("M")

    malignant_probabilities = probabilities[:, malignant_index]

    return {
        "model": best_model,
        "grid": grid,
        "X_train": X_train,
        "X_test": X_test,
        "y_test": y_test,
        "predictions": predictions,
        "probabilities": malignant_probabilities,
        "features": X.columns.tolist()
    }


# ==========================================================
# 5. INITIALIZE APPLICATION
# ==========================================================

st.sidebar.title("🩺 BreastCare AI")

st.sidebar.caption(
    "Breast Cancer Detection Using Machine Learning"
)

st.sidebar.divider()

try:
    df = load_data()
    results = train_model(df)

except Exception as error:
    st.error(f"Unable to initialize application: {error}")
    st.stop()


model = results["model"]
grid = results["grid"]

X_test = results["X_test"]
y_test = results["y_test"]
predictions = results["predictions"]
probabilities = results["probabilities"]
features = results["features"]


# ==========================================================
# 6. EVALUATION METRICS
# ==========================================================

accuracy = accuracy_score(y_test, predictions)

precision = precision_score(
    y_test, predictions, pos_label="M", zero_division=0
)

recall = recall_score(
    y_test, predictions, pos_label="M", zero_division=0
)

f1 = f1_score(
    y_test, predictions, pos_label="M", zero_division=0
)

auc = roc_auc_score(
    (y_test == "M").astype(int),
    probabilities
)


# ==========================================================
# 7. SIDEBAR NAVIGATION
# ==========================================================

st.sidebar.subheader("NAVIGATION")

page = st.sidebar.radio(
    "Select Dashboard Page",
    [
        "Overview",
        "Dataset Explorer",
        "Visualizations",
        "Model Performance",
        "Cancer Prediction"
    ]
)

st.sidebar.divider()

st.sidebar.caption("PROJECT INFORMATION")

st.sidebar.write("Algorithm: SVM")
st.sidebar.write("Kernel: RBF")
st.sidebar.write("Dataset: Wisconsin Diagnostic")
st.sidebar.write("Samples: 569")
st.sidebar.write("Features: 30")

st.sidebar.warning(
    "Educational research application only. "
    "Not intended for medical diagnosis."
)


# ==========================================================
# 8. OVERVIEW PAGE
# ==========================================================

if page == "Overview":

    st.title("Breast Cancer Detection Dashboard")

    st.markdown(
        "### Intelligent classification using Support Vector Machine"
    )

    st.write(
        "This project uses machine learning to classify "
        "breast tumors as Benign or Malignant using "
        "the Wisconsin Breast Cancer Diagnostic dataset."
    )

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Samples",
        f"{len(df):,}"
    )

    col2.metric(
        "Total Features",
        len(features)
    )

    col3.metric(
        "Model Accuracy",
        f"{accuracy * 100:.2f}%"
    )

    col4.metric(
        "ROC-AUC",
        f"{auc:.3f}"
    )

    st.divider()

    st.subheader("Dataset Distribution")

    diagnosis_counts = df["diagnosis"].value_counts()

    distribution = pd.DataFrame({
        "Diagnosis": ["Benign", "Malignant"],
        "Count": [
            diagnosis_counts.get("B", 0),
            diagnosis_counts.get("M", 0)
        ]
    })

    fig = px.pie(
        distribution,
        names="Diagnosis",
        values="Count",
        hole=0.55,
        color="Diagnosis",
        color_discrete_map={
            "Benign": "#42D6C5",
            "Malignant": "#F06B85"
        },
        title="Benign vs Malignant Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Model Information")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Machine Learning Algorithm**")
        st.write("Support Vector Machine (SVM)")
        st.write("Kernel: Radial Basis Function (RBF)")
        st.write("Feature Scaling: StandardScaler")

    with c2:
        st.markdown("**Model Optimization**")
        st.write("GridSearchCV")
        st.write("Cross Validation: 5-fold")
        st.write("Best Parameters:")
        st.json(grid.best_params_)


# ==========================================================
# 9. DATASET EXPLORER
# ==========================================================

elif page == "Dataset Explorer":

    st.title("Dataset Explorer")

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Records", len(df))
    c2.metric("Total Features", len(features))
    c3.metric(
        "Missing Values",
        int(df.isnull().sum().sum())
    )

    st.divider()

    st.subheader("Dataset Preview")

    st.dataframe(
        df,
        use_container_width=True,
        height=400
    )

    st.subheader("Diagnosis Statistics")

    counts = df["diagnosis"].value_counts()

    st.write(f"Benign samples: {counts.get('B', 0)}")
    st.write(f"Malignant samples: {counts.get('M', 0)}")

    st.subheader("Feature Statistics")

    st.dataframe(
        df[features].describe().round(3),
        use_container_width=True
    )


# ==========================================================
# 10. VISUALIZATION PAGE
# ==========================================================

elif page == "Visualizations":

    st.title("Data Visualization")

    st.subheader("Correlation Heatmap")

    corr = df[features].corr()

    fig = px.imshow(
        corr,
        color_continuous_scale="RdBu_r",
        aspect="auto",
        title="Feature Correlation Matrix"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    st.subheader("Feature Distribution")

    selected_feature = st.selectbox(
        "Choose Feature",
        features
    )

    plot_data = df.copy()

    plot_data["Diagnosis"] = plot_data["diagnosis"].map({
        "B": "Benign",
        "M": "Malignant"
    })

    fig = px.histogram(
        plot_data,
        x=selected_feature,
        color="Diagnosis",
        barmode="overlay",
        opacity=0.7,
        color_discrete_map={
            "Benign": "#42D6C5",
            "Malignant": "#F06B85"
        },
        title=f"Distribution of {selected_feature}"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    st.subheader("Feature Relationship")

    col1, col2 = st.columns(2)

    x_feature = col1.selectbox(
        "X Axis",
        features,
        index=0
    )

    y_feature = col2.selectbox(
        "Y Axis",
        features,
        index=1
    )

    fig = px.scatter(
        plot_data,
        x=x_feature,
        y=y_feature,
        color="Diagnosis",
        color_discrete_map={
            "Benign": "#42D6C5",
            "Malignant": "#F06B85"
        },
        title="Feature Comparison"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ==========================================================
# 11. MODEL PERFORMANCE
# ==========================================================

elif page == "Model Performance":

    st.title("SVM Model Performance")

    st.subheader("Evaluation Metrics")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Accuracy",
        f"{accuracy * 100:.2f}%"
    )

    c2.metric(
        "Precision (Malignant)",
        f"{precision * 100:.2f}%"
    )

    c3.metric(
        "Recall (Malignant)",
        f"{recall * 100:.2f}%"
    )

    c4.metric(
        "F1 Score (Malignant)",
        f"{f1 * 100:.2f}%"
    )

    st.divider()

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=["M", "B"]
    )

    fig = px.imshow(
        cm,
        text_auto=True,
        x=["Malignant", "Benign"],
        y=["Malignant", "Benign"],
        color_continuous_scale="Teal",
        labels={
            "x": "Predicted",
            "y": "Actual",
            "color": "Samples"
        },
        title="Actual vs Predicted"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    st.subheader("Classification Report")

    report = classification_report(
        y_test,
        predictions,
        labels=["M", "B"],
        target_names=["Malignant", "Benign"],
        output_dict=True,
        zero_division=0
    )

    st.dataframe(
        pd.DataFrame(report).transpose().round(3),
        use_container_width=True
    )

    st.divider()

    st.subheader("ROC Curve")

    y_binary = (y_test == "M").astype(int)

    fpr, tpr, thresholds = roc_curve(
        y_binary,
        probabilities
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"SVM AUC = {auc:.3f}",
            line=dict(
                color="#42D6C5",
                width=3
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random Classifier",
            line=dict(dash="dash")
        )
    )

    fig.update_layout(
        title="Receiver Operating Characteristic",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    st.subheader("Optimized Hyperparameters")

    st.json(grid.best_params_)

    st.metric(
        "Best Cross-Validation Accuracy",
        f"{grid.best_score_ * 100:.2f}%"
    )


# ==========================================================
# 12. CANCER PREDICTION
# ==========================================================

elif page == "Cancer Prediction":

    st.title("Cancer Prediction")

    st.write(
        "Use the trained SVM model to classify a "
        "sample using its 30 numerical measurements."
    )

    st.warning(
        "This is an educational machine learning demonstration, "
        "not a medical diagnostic tool. Do not use its output "
        "to make healthcare decisions."
    )

    st.divider()

    input_method = st.radio(
        "Choose Prediction Method",
        [
            "Use Existing Dataset Sample",
            "Enter Measurements Manually"
        ]
    )

    input_data = None

    # --------------------------------------
    # EXISTING SAMPLE
    # --------------------------------------

    if input_method == "Use Existing Dataset Sample":

        st.subheader("Select Dataset Sample")

        sample_index = st.selectbox(
            "Choose a sample",
            options=df.index.tolist(),
            format_func=lambda i: (
                f"Sample {i + 1} "
                f"({ 'Malignant' if df.loc[i, 'diagnosis'] == 'M' else 'Benign' })"
            )
        )

        selected_row = df.loc[sample_index]

        input_data = pd.DataFrame(
            [selected_row[features].to_dict()],
            columns=features
        )

        actual_class = selected_row["diagnosis"]

        st.info(
            "Actual dataset label: "
            + (
                "Malignant" if actual_class == "M"
                else "Benign"
            )
        )

        predict_clicked = st.button(
            "Run SVM Prediction",
            type="primary",
            use_container_width=True
        )

    # --------------------------------------
    # MANUAL INPUT
    # --------------------------------------

    else:

        st.subheader("Enter Feature Measurements")

        st.caption(
            "The default values are training-set medians. "
            "Replace them with your intended numerical inputs."
        )

        defaults = results["X_train"].median()

        with st.form("manual_prediction_form"):

            input_values = {}

            for start in range(0, len(features), 3):

                columns = st.columns(3)

                for col, feature in zip(
                    columns,
                    features[start:start + 3]
                ):

                    input_values[feature] = col.number_input(
                        feature,
                        value=float(defaults[feature]),
                        format="%.5f"
                    )

            predict_clicked = st.form_submit_button(
                "Run SVM Prediction",
                type="primary",
                use_container_width=True
            )

        if predict_clicked:

            input_data = pd.DataFrame(
                [input_values],
                columns=features
            )

    # --------------------------------------
    # PREDICTION OUTPUT
    # --------------------------------------

    if predict_clicked and input_data is not None:

        predicted_class = model.predict(input_data)[0]

        predicted_probabilities = model.predict_proba(input_data)[0]

        malignant_index = list(model.classes_).index("M")

        malignant_probability = (
            predicted_probabilities[malignant_index] * 100
        )

        benign_probability = 100 - malignant_probability

        st.divider()

        st.subheader("Prediction Result")

        if predicted_class == "M":

            st.error("Prediction: MALIGNANT")

        else:

            st.success("Prediction: BENIGN")

        c1, c2 = st.columns(2)

        c1.metric(
            "Malignant Model Probability",
            f"{malignant_probability:.2f}%"
        )

        c2.metric(
            "Benign Model Probability",
            f"{benign_probability:.2f}%"
        )

        fig = px.bar(
            x=["Malignant", "Benign"],
            y=[malignant_probability, benign_probability],
            color=["Malignant", "Benign"],
            color_discrete_map={
                "Malignant": "#F06B85",
                "Benign": "#42D6C5"
            },
            labels={
                "x": "Prediction Class",
                "y": "Model Probability (%)"
            },
            title="Model Probability Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.caption(
            "Probabilities are model estimates, not clinical risk "
            "or diagnostic certainty."
        )


# ==========================================================
# 13. FOOTER
# ==========================================================

st.sidebar.divider()

st.sidebar.caption(
    "Breast Cancer Detection | Machine Learning Project"
)

st.sidebar.caption(
    "For educational and research purposes only."
)
