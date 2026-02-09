FROM apache/airflow:3.1.7
ADD requirements.txt .
RUN pip install apache-airflow==3.1.7 -r requirements.txt