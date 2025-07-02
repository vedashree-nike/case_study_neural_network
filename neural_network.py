import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os  # Import os for file path operations
import matplotlib.pyplot as plt

# csv generation
def generate_csv(filename='bin.csv', rows=10000, cols=9):
    """
    Generates a CSV file with random binary data.
    """
    print(f"Generating CSV file '{filename}' with {rows} rows and {cols} columns...")
    data = np.random.randint(0, 2, size=(rows, cols))
    column_names = [f'Col_{i + 1}' for i in range(cols)]
    df = pd.DataFrame(data, columns=column_names)
    df.to_csv(filename, index=False)
    print(f"CSV file '{filename}' generated successfully.")


# implementation of autoencoders
class AE(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc = nn.Sequential(
            nn.Linear(9, 6),
            nn.ReLU(),
            nn.Linear(6, 3),
            nn.ReLU()
        )
        # Decoder
        self.dec = nn.Sequential(
            nn.Linear(3, 6),
            nn.ReLU(),
            nn.Linear(6, 9),
            nn.Sigmoid()  # Sigmoid is good for binary (0/1) reconstruction
        )

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z)


def run_anomaly_detection(csv_filename='bin.csv', num_epochs=30, batch_size=64, learning_rate=1e-3):
    """
    Loads data from CSV, trains an Autoencoder, and performs anomaly detection.
    """
    print(f"\n--- Starting Anomaly Detection with {csv_filename} ---")

    # Load the CSV data
    if not os.path.exists(csv_filename):
        print(f"Error: '{csv_filename}' not found. Please ensure it was generated.")
        return

    try:
        df = pd.read_csv(csv_filename)
        X = df.values.astype(np.float32)  # Convert DataFrame to NumPy array
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return

    # Check if data is empty
    if X.size == 0:
        print("Error: Loaded data is empty. Cannot proceed with training.")
        return

    print(f"Successfully loaded data of shape: {X.shape}")

    dataset = TensorDataset(torch.from_numpy(X))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AE().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Starting training on {device}...")
    for epoch in range(num_epochs):
        total_loss = 0
        for (x_batch,) in loader:
            x = x_batch.to(device)
            x_hat = model(x)
            loss = criterion(x_hat, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(loader.dataset):.6f}')

    # Anomaly Detection
    model.eval()  # Set model to evaluation mode

    print("\nPerforming anomaly detection...")
    with torch.no_grad():
        X_tensor = torch.from_numpy(X).to(device)
        X_hat = model(X_tensor)
        rec_errors = torch.mean((X_hat - X_tensor) ** 2, dim=1).cpu().numpy()

    # Calculate threshold
    th = rec_errors.mean() + 2 * rec_errors.std()  # 2 standard deviations from the mean
    anomaly_idx = np.where(rec_errors > th)[0]

    print(f'\n--- Anomaly Detection Results ---')
    print(f'Total data points: {len(loader.dataset)}')
    print(f'Detected {len(anomaly_idx)} anomalies, with a threshold of: {th:.4f}')

    # Detailed Anomaly Inspection
    if len(anomaly_idx) > 0:
        diffs = torch.abs((X_hat - X_tensor)).cpu().numpy()
        print("\nDetailed errors for the first 5 detected anomaly rows:")
        for i, idx in enumerate(anomaly_idx):
            if i >= 5:  # Limit to top 5 for display
                break
            print(f'Row {idx} (original data): {X[idx]}')
            print(f'Row {idx} (reconstructed data): {X_hat[idx].cpu().numpy()}')
            print(f'Row {idx} errors per column: {diffs[idx]}')
            print('Max error at column (0-indexed):', np.argmax(diffs[idx]))
            print("-" * 50)
    else:
        print("No anomalies detected based on the set threshold.")

    plt.hist(rec_errors, bins=50)
    plt.xlabel('MSE per row')
    plt.ylabel('Count')
    plt.title('Error Distribution')
    plt.show()

# --- Main execution block ---
if __name__ == "__main__":
    csv_file = 'bin.csv'
    generate_csv(csv_file)
    run_anomaly_detection(csv_file)




