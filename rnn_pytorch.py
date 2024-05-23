import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt

# Parameter split_percent defines the ratio of training examples
def get_train_test(url, split_percent=0.8):
    df = pd.read_csv(url, usecols=[1])
    data = np.array(df.values.astype('float32'))
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data).flatten()
    n = len(data)
    # Point for splitting data into train and test
    split = int(n * split_percent)
    train_data = data[:split]
    test_data = data[split:]
    return train_data, test_data, data

# Prepare the input X and target Y
def get_XY(dat, time_steps):
    Y_ind = np.arange(time_steps, len(dat), time_steps)
    Y = dat[Y_ind]
    rows_x = len(Y)
    X = dat[:time_steps * rows_x]
    X = np.reshape(X, (rows_x, time_steps, 1))
    return X, Y

class RNN(nn.Module):
    def __init__(self, input_size, hidden_units, dense_units):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_units, batch_first=True, nonlinearity='tanh')
        self.fc = nn.Linear(hidden_units, dense_units)
    
    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), hidden_units).to(x.device)  # Initial hidden state
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

def print_error(trainY, testY, train_predict, test_predict):
    train_rmse = math.sqrt(mean_squared_error(trainY, train_predict))
    test_rmse = math.sqrt(mean_squared_error(testY, test_predict))
    print(f'Train RMSE: {train_rmse:.3f} RMSE')
    print(f'Test RMSE: {test_rmse:.3f} RMSE')

# Plot the result
def plot_result(trainY, testY, train_predict, test_predict):
    actual = np.append(trainY, testY)
    predictions = np.append(train_predict, test_predict)
    rows = len(actual)
    plt.figure(figsize=(15, 6), dpi=80)
    plt.plot(range(rows), actual)
    plt.plot(range(rows), predictions)
    plt.axvline(x=len(trainY), color='r')
    plt.legend(['Actual', 'Predictions'])
    plt.xlabel('Observation number after given time steps')
    plt.ylabel('Sunspots scaled')
    plt.title('Actual and Predicted Values. The Red Line Separates The Training And Test Examples')
    plt.show()

# Load data
sunspots_url = 'monthly-sunspots.csv'
time_steps = 12
train_data, test_data, data = get_train_test(sunspots_url)
trainX, trainY = get_XY(train_data, time_steps)
testX, testY = get_XY(test_data, time_steps)

# Convert data to PyTorch tensors
trainX = torch.tensor(trainX, dtype=torch.float32)
trainY = torch.tensor(trainY, dtype=torch.float32).unsqueeze(1)
testX = torch.tensor(testX, dtype=torch.float32)
testY = torch.tensor(testY, dtype=torch.float32).unsqueeze(1)

# Create model
input_size = 1
hidden_units = 3
dense_units = 1
model = RNN(input_size, hidden_units, dense_units)

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 20
batch_size = 1

for epoch in range(epochs):
    model.train()
    for i in range(0, len(trainX), batch_size):
        x_batch = trainX[i:i+batch_size]
        y_batch = trainY[i:i+batch_size]

        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Make predictions
model.eval()
with torch.no_grad():
    train_predict = model(trainX).numpy()
    test_predict = model(testX).numpy()

# Print error
print_error(trainY.numpy(), testY.numpy(), train_predict, test_predict)

# Plot result
plot_result(trainY.numpy(), testY.numpy(), train_predict, test_predict)