"""Smoke test for the local Apache Airflow deployment."""

from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator


def validate_environment():
    import os

    print("Airflow smoke test started")
    print(f"Current user ID: {os.getuid()}")
    print("Airflow task execution successful")


with DAG(
    dag_id="pet_breed_smoke_test",
    description="Validate Airflow DAG discovery and task execution",
    start_date=datetime(2026, 10, 1),
    schedule=None,
    catchup=False,
    tags=["pet-breed", "smoke-test"],
) as dag:
    validate = PythonOperator(
        task_id="validate_environment",
        python_callable=validate_environment,
    )
