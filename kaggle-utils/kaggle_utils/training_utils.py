import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd


def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                criterion: nn.Module,
                optimizer: torch.optim.Optimizer,
                num_epochs: int = 100,
                print_interval: int = 10):
    train_losses = []
    val_losses = []
    model.train()

    for epoch in range(num_epochs):
        for features, labels in train_loader:
            outputs = model(features)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, labels in val_loader:
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        train_losses.append(loss.item())
        val_losses.append(avg_val_loss)
        if (epoch + 1) % print_interval == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], \t \
            Training Loss: {loss.item():.4f}, \t \
            Validation Loss: {avg_val_loss:.4f}')

    print('Finish Training')
    return train_losses, val_losses


def split_and_train(X, y, model, num_epochs):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values,
                                  dtype=torch.float32).view(-1, 1)
    X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values,
                                dtype=torch.float32).view(-1, 1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    train_losses, val_losses = train_model(model=model,
                                           train_loader=train_loader,
                                           val_loader=val_loader,
                                           criterion=criterion,
                                           optimizer=optimizer,
                                           num_epochs=num_epochs)
    return model, train_losses, val_losses


def predict_test_data(model: nn.Module,
                      test_df: pd.DataFrame,
                      label_name: str) -> pd.Series:
    test_tensor = torch.tensor(test_df.values, dtype=torch.float32)
    model.eval()

    with torch.no_grad():
        predictions = model(test_tensor)

    test_ids = test_df['Id'].astype('int32')
    predictions_np = predictions.squeeze().numpy()
    predictions_series = pd.Series(predictions_np, name=label_name)
    predictions_series = predictions_series
    submission_df = pd.DataFrame({
        'Id': test_ids,
        label_name: predictions_series
    })
    return submission_df