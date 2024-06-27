import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import mlflow
import mlflow.pytorch
from prometheus_client import start_http_server, Summary, Gauge
import matplotlib.pyplot as plt
from model import CarPriceModel

# Start Prometheus metrics server
start_http_server(8000)

# Create Prometheus metrics to track
REQUEST_TIME = Summary('request_processing_seconds', 'Time spent processing request')
TRAIN_LOSS = Gauge('train_loss', 'Training loss')
VAL_LOSS = Gauge('val_loss', 'Validation loss')
TRAIN_MSE = Gauge('train_mse', 'Training MSE')
VAL_MSE = Gauge('val_mse', 'Validation MSE')

# Load the data
data_train = pd.read_csv("data/train_data.csv")
data_val = pd.read_csv("data/validation_data.csv")

# Split the data into features (X) and target (y)
X_train = data_train.drop("price", axis=1)
y_train = data_train["price"]
X_val = data_val.drop("price", axis=1)
y_val = data_val["price"]

# Identify categorical columns for one-hot encoding
categorical_columns = X_train.select_dtypes(include=['object', 'bool']).columns

# One-hot encode categorical columns
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_train_encoded = pd.DataFrame(encoder.fit_transform(X_train[categorical_columns]))
X_val_encoded = pd.DataFrame(encoder.transform(X_val[categorical_columns]))

# Remove original categorical columns and append the one-hot encoded columns
X_train = X_train.drop(categorical_columns, axis=1).reset_index(drop=True)
X_val = X_val.drop(categorical_columns, axis=1).reset_index(drop=True)
X_train = pd.concat([X_train, X_train_encoded], axis=1)
X_val = pd.concat([X_val, X_val_encoded], axis=1)

# Handle NaNs by filling them with the mean of the column
X_train = X_train.apply(pd.to_numeric, errors='coerce')
X_val = X_val.apply(pd.to_numeric, errors='coerce')
X_train = X_train.fillna(X_train.mean())
X_val = X_val.fillna(X_val.mean())

# Handle infinite values by replacing them with the maximum finite value
X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
X_val.replace([np.inf, -np.inf], np.nan, inplace=True)
X_train = X_train.fillna(X_train.mean())
X_val = X_val.fillna(X_val.mean())

# Convert to numpy arrays
X_train = X_train.values
y_train = y_train.values
X_val = X_val.values
y_val = y_val.values

# Normalize the target variable
y_train_mean = y_train.mean()
y_train_std = y_train.std()
y_train = (y_train - y_train_mean) / y_train_std
y_val = (y_val - y_train_mean) / y_train_std

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize the model
model = CarPriceModel(X_train_tensor.shape[1])

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Reduced learning rate

def calculate_mse(model, loader):
    """
    Calculate mean squared error (MSE) for the model.

    Args:
        model (nn.Module): The neural network model.
        loader (DataLoader): DataLoader for the dataset.

    Returns:
        float: Mean squared error.
    """
    model.eval()
    mse_loss = 0.0
    with torch.no_grad():
        for inputs, targets in loader:
            outputs = model(inputs)
            mse_loss += criterion(outputs, targets).item() * inputs.size(0)
    mse_loss /= len(loader.dataset)
    return mse_loss

# Training and validation
num_epochs = 100
early_stopping_patience = 10
best_val_loss = float('inf')
patience_counter = 0

train_losses = []
val_losses = []
train_mses = []
val_mses = []

# Set the experiment name
mlflow.set_experiment("car_price_prediction")

with mlflow.start_run():
    # Log parameters
    mlflow.log_param("learning_rate", 0.0001)
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("num_epochs", 100)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        train_loss = running_loss / len(train_loader.dataset)
        train_mse = calculate_mse(model, train_loader)
        train_losses.append(train_loss)
        train_mses.append(train_mse)
        
        val_loss = calculate_mse(model, val_loader)
        val_losses.append(val_loss)
        val_mses.append(val_loss)
        
        # Log metrics to mlflow
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("train_mse", train_mse, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_mse", val_loss, step=epoch)

        # Update Prometheus metrics
        TRAIN_LOSS.set(train_loss)
        VAL_LOSS.set(val_loss)
        TRAIN_MSE.set(train_mse)
        VAL_MSE.set(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train MSE: {train_mse:.4f}, Val Loss: {val_loss:.4f}, Val MSE: {val_loss:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'model/car_price_model.pth')
            # Log the model to mlflow
            mlflow.pytorch.log_model(model, "car_price_model")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered")
                break

    # Plot training history
    plt.plot(train_losses, label='train_loss')
    plt.plot(val_losses, label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("model/loss_plot.png")
    mlflow.log_artifact("model/loss_plot.png")

    plt.figure()
    plt.plot(train_mses, label='train_mse')
    plt.plot(val_mses, label='val_mse')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.savefig("model/mse_plot.png")
    mlflow.log_artifact("model/mse_plot.png")

