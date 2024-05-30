FROM python:3.10-slim

# mlflow 2.12.1 should be already in the requirements txt that sets up mage
RUN pip install mlflow==2.12.1

EXPOSE 5000

# sets up mlflow server to look at locally
CMD [ \
    "mlflow", "server", \
    "--backend-store-uri", "sqlite:///home/mlflow/mlflow.db", \
    "--host", "0.0.0.0", \
    "--port", "5000" \
]