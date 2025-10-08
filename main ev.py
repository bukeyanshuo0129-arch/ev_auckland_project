import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib.ticker import MaxNLocator
# reproducibility
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ---------------- Dataset ----------------
class EVChargingDataset:
    def __init__(self):
        # duration (only for stations that exist)
        self.duration_20 = pd.read_excel('ev_duration_data.xlsx', sheet_name='duration_20', index_col=0)
        self.duration_24 = pd.read_excel('ev_duration_data.xlsx', sheet_name='duration_24', index_col=0)
        self.duration_15 = pd.read_excel('ev_duration_data.xlsx', sheet_name='duration_15', index_col=0)
        self.duration_16 = pd.read_excel('ev_duration_data.xlsx', sheet_name='duration_16', index_col=0)

        # loads
        self.load_20 = pd.read_excel('ev_load_data1.xlsx', sheet_name='node_20', index_col=0)
        self.load_24 = pd.read_excel('ev_load_data1.xlsx', sheet_name='node_24', index_col=0)
        self.load_15 = pd.read_excel('ev_load_data1.xlsx', sheet_name='node_15', index_col=0)
        self.load_16 = pd.read_excel('ev_load_data1.xlsx', sheet_name='node_16', index_col=0)
        self.load_1  = pd.read_excel('ev_load_data1.xlsx', sheet_name='node_1',  index_col=0)
        self.load_8  = pd.read_excel('ev_load_data1.xlsx', sheet_name='node_8',  index_col=0)

        # rain (two columns: time, rain) or one column indexed by time; we use first value
        self.rain = pd.read_excel('ev_load_data1.xlsx', sheet_name='rain', index_col=0)

        self.preprocess_data()

    def _clean_and_scale(self, df):
        df = df.copy()
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        # remove zero-variance columns (avoid NaN after standardization)
        keep = df.var() > 0
        if keep.sum() == 0:
            # fallback: if all columns zero variance, keep them (scaler will be identity-like)
            keep[:] = True
        df = df.loc[:, keep]
        scaler = StandardScaler()
        X = scaler.fit_transform(df.values)
        X = np.nan_to_num(X)  # safety
        return pd.DataFrame(X, index=df.index, columns=df.columns), scaler

    def preprocess_data(self):
        self.all_durations_raw = {
            'station_20': self.duration_20,
            'station_24': self.duration_24,
            'station_15': self.duration_15,
            'station_16': self.duration_16
        }
        self.all_loads_raw = {
            'station_20': self.load_20,
            'station_24': self.load_24,
            'station_15': self.load_15,
            'station_16': self.load_16,
            'station_1':  self.load_1,
            'station_8':  self.load_8
        }

        # scale loads
        self.all_loads, self.scalers = {}, {}
        for k, df in self.all_loads_raw.items():
            df_s, sc = self._clean_and_scale(df)
            self.all_loads[k], self.scalers[k] = df_s, sc

        # scale durations
        self.all_durations, self.duration_scalers = {}, {}
        for k, df in self.all_durations_raw.items():
            df_s, sc = self._clean_and_scale(df)
            self.all_durations[k], self.duration_scalers[k] = df_s, sc

        # align rain to numeric 0/1 vector
        rain_series = self.rain.iloc[:, 0] if self.rain.shape[1] >= 1 else self.rain.squeeze()
        rain_series = rain_series.replace([np.inf, -np.inf], np.nan).fillna(0).astype(float)
        # clamp to {0,1} if user data is indicator
        rain_series = rain_series.clip(0, 1)
        self.rain = pd.DataFrame({'rain': rain_series.values}, index=rain_series.index)

    def create_graph_data(self, sequence_length=7, predict_ahead=1):
        graphs = []
        station_keys = list(self.all_loads.keys())
        num_stations = len(station_keys)

        # fully connected undirected edges (both directions)
        edges = [[i, j] for i in range(num_stations) for j in range(num_stations) if i != j]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        T = len(self.rain)
        for day in range(sequence_length, T - predict_ahead):
            node_features = []
            for st in station_keys:
                load_feat, dur_feat = [], []
                for t in range(sequence_length):
                    idx = day - sequence_length + t
                    load_feat.extend(self.all_loads[st].iloc[idx].values)  # 24 values
                    if st in self.all_durations:
                        dur_feat.extend(self.all_durations[st].iloc[idx].values)  # 24 values
                    else:
                        dur_feat.extend([0.0] * 24)
                node_features.append(load_feat + dur_feat)  # len = 24*seq*2

            node_features = np.nan_to_num(np.array(node_features, dtype=np.float32))
            x = torch.tensor(node_features, dtype=torch.float)

            # label: next day's 24h loads of all stations
            y = []
            fut = day + predict_ahead
            for st in station_keys:
                y.extend(self.all_loads[st].iloc[fut].values)  # 24
            y = torch.tensor(np.array(y, dtype=np.float32), dtype=torch.float)

            # weather features
            rh = [float(self.rain.iloc[day - sequence_length + t, 0]) for t in range(sequence_length)]
            fr = float(self.rain.iloc[fut, 0])
            rain_history = torch.tensor([rh], dtype=torch.float)         # [1, seq]
            future_rain  = torch.tensor([[fr]], dtype=torch.float)       # [1, 1]

            # sanity checks
            if torch.isnan(x).any() or torch.isinf(x).any():
                # skip bad sample
                continue
            if torch.isnan(y).any() or torch.isinf(y).any():
                continue

            graphs.append(
                Data(
                    x=x,
                    edge_index=edge_index,
                    y=y,
                    rain_history=rain_history,
                    future_rain=future_rain
                    )
            )
        return graphs

# ---------------- Model ----------------
class GCNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_stations, sequence_length):
        super().__init__()
        self.num_stations = num_stations
        self.sequence_length = sequence_length

        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        self.weather_fc = nn.Sequential(
            nn.Linear(sequence_length + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.out_fc = nn.Sequential(
            nn.Linear(hidden_dim * num_stations + hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, output_dim)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        rain_history, future_rain = data.rain_history, data.future_rain  # [1,seq], [1,1]

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        # clamp features to avoid exploding values
        x = torch.clamp(x, -10.0, 10.0)

        node_features = x.reshape(1, -1)  # [1, num_stations*hidden_dim]

        weather_input = torch.cat([rain_history, future_rain], dim=1)  # [1, seq+1]
        weather_features = self.weather_fc(weather_input)               # [1, hidden_dim]

        combined = torch.cat([node_features, weather_features], dim=1)  # [1, num_stations*h + h]
        out = self.out_fc(combined)                                     # [1, output_dim]
        return out.squeeze(0)                                           # [output_dim]

# ---------------- Train ----------------
def train_model(model, train_loader, val_loader, num_epochs=100, lr=1e-5):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        tr = 0.0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)              # [output_dim]
            loss = criterion(out, data.y)  # data.y: [output_dim]
            if torch.isnan(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            tr += float(loss.item())
        train_losses.append(tr / max(1, len(train_loader)))

        model.eval()
        va = 0.0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                out = model(data)
                loss = criterion(out, data.y)
                va += float(loss.item())
        val_losses.append(va / max(1, len(val_loader)))

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}]  Train: {train_losses[-1]:.6f} ")

    return train_losses, val_losses

# ---------------- Predict ----------------
def predict_future(model, dataset, rain_condition, sequence_length=7):
    model.eval()
    station_keys = list(dataset.all_loads.keys())
    num_stations = len(station_keys)

    # build one graph using the latest sequence_length days
    node_features = []
    for st in station_keys:
        load_feat, dur_feat = [], []
        for t in range(sequence_length):
            idx = len(dataset.rain) - sequence_length + t
            load_feat.extend(dataset.all_loads[st].iloc[idx].values)
            if st in dataset.all_durations:
                dur_feat.extend(dataset.all_durations[st].iloc[idx].values)
            else:
                dur_feat.extend([0.0] * 24)
        node_features.append(load_feat + dur_feat)
    node_features = np.nan_to_num(np.array(node_features, dtype=np.float32))
    x = torch.tensor(node_features, dtype=torch.float)

    # edges
    edges = [[i, j] for i in range(num_stations) for j in range(num_stations) if i != j]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # weather
    rh = [float(dataset.rain.iloc[len(dataset.rain)-sequence_length + t, 0]) for t in range(sequence_length)]
    rain_history = torch.tensor([rh], dtype=torch.float)           # [1, seq]
    future_rain  = torch.tensor([[float(rain_condition)]], dtype=torch.float)  # [1,1]

    data = Data(x=x, edge_index=edge_index, y=torch.zeros(num_stations*24),
                rain_history=rain_history, future_rain=future_rain).to(device)

    with torch.no_grad():
        pred = model(data).cpu().numpy().reshape(num_stations, 24)

    # inverse transform back to original scale
    for i, st in enumerate(station_keys):
        pred[i] = dataset.scalers[st].inverse_transform(pred[i].reshape(1, -1)).flatten()

    return pred, station_keys

# ---------------- Main ----------------
def main():
    print("Loading and preprocessing data")
    dataset = EVChargingDataset()

    print("Creating graph data")
    graphs = dataset.create_graph_data(sequence_length=7, predict_ahead=1)
    if len(graphs) == 0:
        raise RuntimeError("No valid graphs were created. Check your data lengths and NaN handling.")

    train_graphs, test_graphs = train_test_split(graphs, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_graphs, batch_size=1, shuffle=True)
    test_loader  = DataLoader(test_graphs,  batch_size=1, shuffle=False)

    num_stations = len(dataset.all_loads)
    sequence_length = 7
    input_dim  = 24 * sequence_length * 2
    hidden_dim = 32
    output_dim = 24 * num_stations

    model = GCNNModel(input_dim, hidden_dim, output_dim, num_stations, sequence_length).to(device)

    print("Training model")
    train_losses, val_losses = train_model(model, train_loader, test_loader, num_epochs=100, lr=1e-5)

    # plot losses
    plt.figure(figsize=(8,4))
    plt.plot(train_losses, label='Train')
    #plt.plot(val_losses, label='Val')
    plt.xlabel('Epoch'); plt.ylabel('MSE'); plt.title('Training Curve'); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig('training_loss.png'); plt.show()

    print("Predicting future load")
    rainy_pred, station_keys = predict_future(model, dataset, rain_condition=1, sequence_length=sequence_length)
    ev_pred, _ = predict_future(model, dataset, rain_condition=0, sequence_length=sequence_length)

    # plot predictions
    fig, axes = plt.subplots(2, 3, figsize=(14,9))
    axes = axes.flatten()
    for i, st in enumerate(station_keys):
        #axes[i].plot(rainy_pred[i], label='Rainy')
        axes[i].plot(ev_pred[i],  label='Predict')
        axes[i].set_title(st); 
        axes[i].set_xlabel('Hour'); 
        axes[i].set_ylabel('Load'); 
        axes[i].grid(True); 
        axes[i].legend()
        axes[i].set_ylim(bottom=0)
        axes[i].xaxis.set_major_locator(MaxNLocator(nbins=6))
    plt.tight_layout(); plt.savefig('predictions_comparison.png'); plt.show()

    torch.save(model.state_dict(), 'gcnn_ev_charging_model.pth')
    print("Model saved as gcnn_ev_charging_model.pth")
    # ---------------- Save predictions to Excel ----------------
    pred_df_rainy = pd.DataFrame(rainy_pred.T, columns=station_keys)
    pred_df_rainy.insert(0, 'Hour', np.arange(24))
    
    pred_df_dry = pd.DataFrame(ev_pred.T, columns=station_keys)
    pred_df_dry.insert(0, 'Hour', np.arange(24))
    
    with pd.ExcelWriter('predicted_loads.xlsx') as writer:
        pred_df_rainy.to_excel(writer, sheet_name='Rainy', index=False)
        pred_df_dry.to_excel(writer, sheet_name='Non-Rainy', index=False)
    
    print("Predictions saved to predicted_loads.xlsx")



if __name__ == "__main__":
    main()
