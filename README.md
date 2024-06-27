# Car Price Prediction

This repository contains code for training a deep neural network to predict car prices using PyTorch. The project includes data preprocessing, model training, and monitoring using Prometheus and mlflow.

## Project Structure

.
├── data
│ ├── train_data.csv
│ └── validation_data.csv
├── model
│ ├── model.py
│ ├── train.py
│── requirements.txt
├── .github
│ └── workflows
│ └── ci.yml
└── README.md



## How to Run

1. **Clone the repository:**

```sh
git clone https://github.com/your-username/car-price-prediction.git
cd car-price-prediction
```sh

Install dependencies:

pip install -r requirements.txt


Run the training script:
python model/train.py

Start the mlflow UI:
mlflow ui


Access the mlflow UI:
Open the link provided after running mlflow ui (usually http://127.0.0.1:5000).

CI/CD Pipeline

The project uses GitHub Actions for CI/CD. The pipeline is defined in .github/workflows/ci.yml and includes steps to:

Install dependencies
Run the training script
Upload the trained model as an artifact
The pipeline is triggered on every push and pull request to the main branch.

Monitoring

Prometheus metrics server is set up to monitor training and validation metrics in real-time.


```sh
from prometheus_client import start_http_server, Summary, Gauge

# Start Prometheus metrics server
start_http_server(8000)

```sh

