

import datetime
import streamlit as st
from sqlalchemy import create_engine
from prefect import task, flow
from prefect.schedules import IntervalSchedule
from evidently.report import Report
from evidently.test_suite import TestSuite
from evidently.test_preset import DataStabilityTestPreset
from evidently.ui.dashboards import Dashboard
from evidently.ui.workspace import Workspace
from evidently.options import ColorOptions
import plotly.graph_objects as go
from evidently.model_profile import ModelProfile
from evidently.model_profile.sections import (
    ModelDriftProfileSection,
    ModelPerformanceProfileSection,
)
from evidently.pipeline.column_mapping import ColumnMapping


@task
def run_test_suite(reference_data, current_data):
    """Run the data stability test suite."""
    test_suite = TestSuite(tests=[DataStabilityTestPreset()])
    test_suite.run(reference_data=reference_data, current_data=current_data)
    return test_suite


@task
def save_results(report, test_suite, reference_data, current_data):
    """Save the results to the dashboard and database."""
    # Create and configure the dashboard with report and test suite
    dashboard = Dashboard(tabs=[report, test_suite])
    workspace = Workspace.create("my_monitoring_project")
    project = workspace.create_project("model_monitoring")
    project.dashboard = dashboard

    # Add a snapshot to the project
    snapshot = project.add_snapshot(
        f"Snapshot {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        reference_data,
        current_data,
    )
    snapshot.run()
    project.save()

    # Collect results for saving to the database
    results = {
        "timestamp": datetime.datetime.now(),
        "data_drift_score": report.get_metric("DataDriftPreset")
        .get_result()
        .current_drift_score,
        "data_quality_score": report.get_metric("DataQualityPreset")
        .get_result()
        .current_quality_score,
        "target_drift_score": report.get_metric("TargetDriftPreset")
        .get_result()
        .current_drift_score,
        "data_stability_score": test_suite.get_test("DataStabilityTestPreset")
        .get_result()
        .status,
    }

    # Save the results to the database
    pd.DataFrame([results]).to_sql(
        "monitoring_results", engine, if_exists="append", index=False
    )


# Streamlit app
def main():
    st.set_page_config(page_title="MLOps Monitoring Dashboard", layout="wide")
    st.title("MLOps Monitoring Dashboard")

    uploaded_file = st.file_uploader("Upload your CSV file to test", type=["csv"])

    reference_data, current_data = fetch_data(uploaded_file)

    if st.button("Run Monitoring Flow"):
        with st.spinner("Running monitoring flow..."):
            data_drift_report, test_suite, model_drift_report = monitoring_flow(
                uploaded_file

        with col3:
            st.subheader("Model Drift")
            model_drift_score = (
                model_drift_report.get_section("model_drift").get_result().drift_score
            )
            st.plotly_chart(
                create_metric_gauge(model_drift_score, "Model Drift Score"),
                use_container_width=True,
            )

        # Model Performance Drift
        performance_drift = model_drift_report.get_section("performance").get_result()
        st.subheader("Model Performance Drift")
        st.write(f"Current ROC AUC: {performance_drift.current_roc_auc:.4f}")
        st.write(f"Reference ROC AUC: {performance_drift.reference_roc_auc:.4f}")
        st.write(f"Performance Drift: {performance_drift.drift_score:.4f}")

        # Test suite results
        st.subheader("Test Suite Results")
        st.write(
            f"Data Stability Test: {test_suite.get_test('DataStabilityTestPreset').get_result().status}"
        )


        # Option to download the full HTML report
        data_drift_report.save_html("full_report.html")
        with open("full_report.html", "rb") as file:
            st.download_button(
                label="Download Full Report",
                data=file,
                file_name="full_report.html",
                mime="text/html",
            )


import pandas as pd
import joblib
import json
import streamlit as st
from sqlalchemy import create_engine
import plotly.graph_objects as go
from prefect import task, flow

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import (
    ColumnDriftMetric,
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
)
from evidently.metric_preset import (
    DataDriftPreset,
    DataQualityPreset,
    TargetDriftPreset,
)

from ev
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection, CatTargetDriftProfileSection

# from evidently.profile import ModelProfile
# from evidently.profile.sections import ModelDriftProfileSection, ModelPerformanceProfileSection
# from evidently import profile
# from profile import model_profile
# from profile.sections import ModelDriftProfileSection, ModelPerformanceProfileSection

import boto3

s3 = boto3.client(
    "s3",
    endpoint_url="http://localstack:4566",
    aws_access_key_id="test",
    aws_secret_access_key="test",
    region_name="us-east-1",
)

bucket_name = "risk-bucket"
model_file = "s3_risk_model.pkl"
transform_file = "s3_transformer.pkl"


# download model from S3
def download_model_from_s3():
    with open(model_file, "wb") as f:
        s3.download_fileobj(bucket_name, model_file, f)

    return joblib.load(model_file)


def download_transformer_from_s3():
    with open(transform_file, "wb") as f:
        s3.download_fileobj(bucket_name, transform_file, f)

    return joblib.load(transform_file)


def _load_s3():
    model = download_model_from_s3()
    transformer = download_transformer_from_s3()

    return model, transformer

# Database setup
engine = create_engine("sqlite:///monitoring_results.db")


# Helper functions
def create_metric_gauge(value, title):
    """Create a gauge chart to visualize scores."""
    return go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": title},
            gauge={
                "axis": {"range": [None, 1]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 0.3], "color": "red"},
                    {"range": [0.3, 0.7], "color": "yellow"},
                    {"range": [0.7, 1], "color": "green"},
                ],
            },
        )
    )


def inspect_columns(reference_data, current_data):
    st.write("Reference Data Columns:", reference_data.columns)
    st.write("Current Data Columns:", current_data.columns)


def reference_data():
    reference_data = pd.read_csv("../data/credit.csv", index_col=0)
    return reference_data


def current_data(uploaded_file=None):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file, index_col=0)
    else:
        return reference_data().sample(frac=0.2, random_state=42)


@task
def fetch_data(uploaded_file=None):
    reference = reference_data()
    current = current_data(uploaded_file)
    return reference, current


@task
def generate_drift_report(reference_data, current_data):
    column_mapping = ColumnMapping()
    column_mapping.target = "loan_status"

    # Set numerical and categorical features based on data types in reference_data
    column_mapping.numerical_features = reference_data.select_dtypes(
        include=["float64", "int64"]
    ).columns.tolist()
    column_mapping.categorical_features = reference_data.select_dtypes(
        include=["object"]
    ).columns.tolist()

    if "loan_status" in column_mapping.numerical_features:
        column_mapping.numerical_features.remove("loan_status")
    if "loan_status" in column_mapping.categorical_features:
        column_mapping.categorical_features.remove("loan_status")

    report = Report(
        metrics=[
            ColumnDriftMetric(column_name="loan_status"),
            DatasetDriftMetric(),
            DatasetMissingValuesMetric(),
            DataDriftPreset(),
            TargetDriftPreset(),
        ]
    )

    report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping,
    )
    return report



    # Set up column mapping for target and prediction columns
    column_mapping = ColumnMapping()
    column_mapping.target = "loan_status"
    column_mapping.prediction = y

    # Create model profile with drift and performance sections
    model_profile = ModelProfile(
        sections=[ModelDriftProfileSection(), ModelPerformanceProfileSection()]
    )
    model_profile.calculate(reference_data, current_data, column_mapping=column_mapping)

    return model_profile

@task
def generate_model_drift_report(reference_data, current_data):
    # Download model/ transformer
    model, transformer = _load_s3()
    
    # Preparing features and target
    x = current_data.drop(columns=["loan_status"], axis=1)
    y = current_data["loan_status"]


    processed = transformer.transform(x)
    predictions = model.predict(processed)
    
    
    # Set up column mapping for target and prediction columns
    column_mapping = ColumnMapping()
    column_mapping.target = "loan_status"
    column_mapping.prediction = "prediction"

    # Add predictions to the current data
    current_data['prediction'] = predictions
    
    # Create model profile with drift and performance sections
    model_profile = profile.ModelProfile(
        sections=[ModelDriftProfileSection(), ModelPerformanceProfileSection()]
    )
    
    # Calculate the profile with drift and performance analysis
    model_profile.calculate(reference_data, current_data, column_mapping=column_mapping)
    
    return model_profile



@flow
def monitoring_flow(uploaded_file=None):
    reference_data, current_data = fetch_data(uploaded_file)
    drift_report = generate_drift_report(reference_data, current_data)
    model_drift_report = generate_model_drift_report(reference_data, current_data)
    return drift_report, model_drift_report


def handle_alerts(result, drift_report):
    alerts = []
    if result["metrics"][0]["result"]["drift_score"] > 0.1:
        alerts.append("ALERT: Significant data drift detected!")

    if alerts:
        for alert in alerts:
            st.error(alert)

    if result["metrics"][1]["result"]["dataset_drift"] < 0.9:
        alerts.append("ALERT: Data quality below threshold!")

    if alerts:
        for alert in alerts:
            st.error(alert)

    # Handle test suite alerts
    if drift_report:
        drift_report_json = json.loads(drift_report.json())
        alerts = [
            section
            for section in drift_report_json["sections"]
            if section["type"] == "text" and "ALERT" in section["content"]
        ]

        if alerts:
            st.subheader("Alerts")
            for alert in alerts:
                st.error(alert["content"])


def display_tabs(result, reference_data, current_data, model_drift_report):
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Columns", "Data Drift", "Data Quality", "Model Drift"]
    )

    with tab1:
        st.subheader("Inspect columns")
        inspect_columns(reference_data, current_data)

    with tab2:
        st.subheader("Data Drift")
        data_drift_score = result["metrics"][0]["result"]["drift_score"]
        st.plotly_chart(
            create_metric_gauge(data_drift_score, "Data Drift Score"),
            use_container_width=True,
        )

    with tab3:
        st.subheader("Data Quality")
        data_quality = result["metrics"][1]["result"]["dataset_drift"]
        status = 1 if data_quality else 0
        st.plotly_chart(
            create_metric_gauge(status, "Data Quality"), use_container_width=True
        )

    with tab4:
        st.subheader("Model Drift")
        # Display model drift
        # st.plotly_chart(
        #     create_metric_gauge(
        #         model_drift_report.metrics[0].value, "Model Drift Score"
        #     ),
        #     use_container_width=True,
        # )

        model_drift_score = model_drift_report.sections[0].metrics[0].value
        st.plotly_chart(
            create_metric_gauge(model_drift_score, "Model Drift Score"),
            use_container_width=True,
        )


def main():
    st.set_page_config(page_title="Monitoring Dashboard", layout="wide")
    st.title("Monitoring Dashboard")

    uploaded_file = st.file_uploader("Upload your CSV file to test", type=["csv"])

    if st.button("Run Monitoring Flow"):
        with st.spinner("Running monitoring flow..."):
            drift_report, model_drift_report = monitoring_flow(uploaded_file)
            reference_data, current_data = fetch_data(uploaded_file=uploaded_file)
            result = drift_report.as_dict()

            st.success("Monitoring flow completed!")

        # Handle alerts
        handle_alerts(result, drift_report)

        # Display tabs with the results
        display_tabs(result, reference_data, current_data, model_drift_report)


if __name__ == "__main__":
    main()
