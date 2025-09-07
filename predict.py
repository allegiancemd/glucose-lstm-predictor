import argparse
import math
import os
from dataclasses import dataclass
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# --------------------------
# Utils
# --------------------------
def clarke_error_grid(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Very lightweight Clarke Error Grid binning for summary counts.
    For formal/regulatory analysis, use a validated implementation.
    """
    A = B = C = D = E = 0
    for ref, est in zip(y_true, y_pred):
        if ref <= 70:
            if est <= 70: A += 1
            elif est <= 180: B += 1
            else: E += 1
        elif ref <= 180:
            if abs(est - ref) <= 0.2 * ref: A += 1
            elif (ref < 70 and est >= 180) or (ref >= 180 and est < 70): E += 1
            elif (ref >= 70 and est <= 70) or (ref <= 180 and est >= 180): B += 1
            else: B += 1
        else:  # ref > 180
            if est >= 180: A += 1 if abs(est - ref) <= 0.2 * ref else B + 0
            elif est <= 70: E += 1
            else: B += 1
    total = max(1, A + B + C + D + E)
    return {"A": A/total, "B": B/total, "C": C/total, "D": D/total, "E": E/total}

def iob_simple(units: pd.Series, tau_min: int = 300) -> pd.Series:
    """
    Exponential insulin-on-board proxy (very simplified).
    tau_min: decay constant in minutes (e.g., 300 = 5 hours)
    """
    # 5-min steps
    alpha = math.exp(-5.0 / tau_min)
    out = []
    acc = 0.0
    for u in units.fillna(0.0):
        acc = acc * alpha + u
        out.append(acc)
    return pd.Series(out, index=units.index)

def cob_simple(carbs: pd.Series, tau_min: int = 120) -> pd.Series:
    """
    Exponential carbs-on-board proxy (very simplified).
    tau_min: decay constant in minutes (e.g., 120 = 2 hours)
    """
    alpha = math.exp(-5.0 / tau_min)
    out = []
    acc = 0.0
    for c in carbs.fillna(0.0):
        acc = acc * alpha + c
        out.append(acc)
    return pd.Series(out, index=carbs.index)

def add_time_features(ts: pd.Series) -> pd.DataFrame:
    minutes = ts.dt.hour * 60 + ts.dt.minute
    sin_t = np.sin(2 * np.pi * minutes / (24 * 60))
    cos_t = np.cos(2 * np.pi * minutes / (24 * 60))
    dow = ts.dt.dayofweek
    sin_w = np.sin(2 * np.pi * dow / 7.0)
    cos_w = np.cos(2 * np.pi * dow / 7.0)
    return pd.DataFrame({"tod_sin": sin_t, "tod_cos": cos_t, "dow_sin": sin_w, "dow_cos": cos_w})

# --------------------------
# Data pipeline
# --------------------------
def load_and_engineer(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Basic cleanup
    if "timestamp" not in df.columns:
        raise ValueError("CSV must include a 'timestamp' column.")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    # Expect 5-min cadence; if not, resample
    df = df.set_index("timestamp").resample("5T").mean().interpolate(limit=6, limit_direction="both").reset_index()

    # Fill missing expected columns with zeros
    for col in ["cgm_mgdl", "bolus_units", "basal_u_per_hr", "carbs_g", "protein_g", "fat_g"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].astype(float).fillna(0.0)

    # Simple IOB/COB features
    df["iob"] = iob_simple(df["bolus_units"] + (df["basal_u_per_hr"] / 12.0))  # basal per 5 min
    df["cob"] = cob_simple(df["carbs_g"])

    # Macronutrient ratios
    total_macro = df[["carbs_g", "protein_g", "fat_g"]].sum(axis=1).replace(0, np.nan)
    df["carb_frac"] = (df["carbs_g"] / total_macro).fillna(0.0)
    df["prot_frac"] = (df["protein_g"] / total_macro).fillna(0.0)
    df["fat_frac"]  = (df["fat_g"] / total_macro).fillna(0.0)

    # Time features
    time_feats = add_time_features(df["timestamp"])
    df = pd.concat([df, time_feats], axis=1)

    # Lag features (helpful for dynamics)
    for lag in [1, 2, 3, 6, 12]:  # 5,10,15,30,60 min
        df[f"cgm_lag_{lag}"] = df["cgm_mgdl"].shift(lag).bfill()

    return df

@dataclass
class WindowSpec:
    lookback: int = 36     # 3 hours @ 5 min
    horizon: int = 6       # 30 min ahead (single step) OR steps for multi-step
    stride: int = 1
    multi_step: bool = False

class GlucoseDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_cols: List[str], target_col: str,
                 win: WindowSpec, scaler: Optional[StandardScaler] = None):
        self.df = df.copy()
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.win = win

        X = self.df[feature_cols].values.astype(np.float32)
        y = self.df[target_col].values.astype(np.float32)

        # fit/transform scaler outside for train, inside here for val/test
        self.scaler = scaler
        if self.scaler is not None:
            X = self.scaler.transform(X)

        self.X_seq, self.y_seq = self._build_windows(X, y)

    def _build_windows(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        L = X.shape[0]
        lb, hz, st = self.win.lookback, self.win.horizon, self.win.stride
        X_seq = []
        y_seq = []
        end = L - (lb + hz) + 1
        for i in range(0, max(0, end), st):
            X_seq.append(X[i:i+lb])
            if self.win.multi_step:
                y_seq.append(y[i+lb:i+lb+hz])
            else:
                y_seq.append(y[i+lb+hz-1])  # predict the last step of horizon
        return np.array(X_seq), np.array(y_seq)

    def __len__(self):
        return self.X_seq.shape[0]

    def __getitem__(self, idx):
        X = self.X_seq[idx]
        y = self.y_seq[idx]
        return torch.from_numpy(X), torch.from_numpy(y)

# --------------------------
# Model
# --------------------------
class LSTMForecaster(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 128, num_layers: int = 2,
                 dropout: float = 0.1, multi_step: bool = False, horizon: int = 6):
        super().__init__()
        self.multi_step = multi_step
        self.horizon = horizon
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, horizon if multi_step else 1)
        )

    def forward(self, x):
        # x: [B, T, F]
        out, _ = self.lstm(x)
        # Use last time step's hidden state
        h_last = out[:, -1, :]
        y = self.head(h_last)
        return y

# --------------------------
# Training / Evaluation
# --------------------------
def train_one_epoch(model, dl, loss_fn, opt, device):
    model.train()
    total = 0.0
    n = 0
    for X, y in dl:
        X = X.to(device)
        y = y.to(device)
        opt.zero_grad()
        pred = model(X).squeeze(-1)
        loss = loss_fn(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item() * X.size(0)
        n += X.size(0)
    return total / max(1, n)

@torch.no_grad()
def evaluate(model, dl, device, multi_step: bool):
    model.eval()
    preds = []
    trues = []
    for X, y in dl:
        X = X.to(device)
        pred = model(X).cpu().numpy()
        preds.append(pred)
        trues.append(y.numpy())
    y_true = np.concatenate(trues, axis=0)
    y_pred = np.concatenate(preds, axis=0)
    if not multi_step:
        # shapes: [N], [N,1] or [N]
        y_pred = y_pred.reshape(-1)
    rmse = mean_squared_error(y_true.reshape(-1), y_pred.reshape(-1), squared=False)
    mae = mean_absolute_error(y_true.reshape(-1), y_pred.reshape(-1))
    ceg = clarke_error_grid(y_true.reshape(-1), y_pred.reshape(-1))
    return rmse, mae, ceg, y_true, y_pred

# --------------------------
# Main
# --------------------------
def main(args):
    df = load_and_engineer(args.data)

    # Feature set
    feature_cols = [
        "cgm_mgdl",
        "bolus_units", "basal_u_per_hr",
        "carbs_g", "protein_g", "fat_g",
        "iob", "cob",
        "carb_frac", "prot_frac", "fat_frac",
        "tod_sin", "tod_cos", "dow_sin", "dow_cos",
        "cgm_lag_1", "cgm_lag_2", "cgm_lag_3", "cgm_lag_6", "cgm_lag_12",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]
    target_col = "cgm_mgdl"

    # Train/val/test split by time (e.g., 70/15/15)
    n = len(df)
    i1 = int(0.7 * n); i2 = int(0.85 * n)
    df_train = df.iloc[:i1].reset_index(drop=True)
    df_val   = df.iloc[i1:i2].reset_index(drop=True)
    df_test  = df.iloc[i2:].reset_index(drop=True)

    # Scale features using train stats only
    scaler = StandardScaler().fit(df_train[feature_cols].values.astype(np.float32))

    win = WindowSpec(lookback=args.lookback, horizon=args.horizon,
                     stride=args.stride, multi_step=args.multi_step)

    ds_train = GlucoseDataset(df_train, feature_cols, target_col, win, scaler)
    ds_val   = GlucoseDataset(df_val,   feature_cols, target_col, win, scaler)
    ds_test  = GlucoseDataset(df_test,  feature_cols, target_col, win, scaler)

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, drop_last=True)
    dl_val   = DataLoader(ds_val,   batch_size=args.batch_size, shuffle=False)
    dl_test  = DataLoader(ds_test,  batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMForecaster(n_features=len(feature_cols), hidden_size=args.hidden,
                           num_layers=args.layers, dropout=args.dropout,
                           multi_step=args.multi_step, horizon=args.horizon).to(device)

    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, dl_train, loss_fn, opt, device)
        rmse, mae, ceg, _, _ = evaluate(model, dl_val, device, args.multi_step)
        val_metric = rmse
        if val_metric < best_val:
            best_val = val_metric
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        print(f"Epoch {epoch:03d} | train_loss={tr_loss:.3f} | val_RMSE={rmse:.2f} | val_MAE={mae:.2f} | Clarke A={ceg['A']:.2f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final test
    rmse, mae, ceg, y_true, y_pred = evaluate(model, dl_test, device, args.multi_step)
    print("\n=== Test Results ===")
    print(f"RMSE: {rmse:.2f} mg/dL")
    print(f"MAE : {mae:.2f} mg/dL")
    print(f"Clarke A-zone fraction: {ceg['A']:.2f}")
    # Save predictions for plotting outside
    np.savez("predictions.npz", y_true=y_true, y_pred=y_pred)

    # Save model
    torch.save(model.state_dict(), "lstm_glucose.pt")
    with open("feature_cols.txt", "w") as f:
        for c in feature_cols: f.write(c + "\n")
    print("Saved: lstm_glucose.pt, predictions.npz, feature_cols.txt")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True, help="Path to CSV with 5-min time series.")
    p.add_argument("--lookback", type=int, default=36, help="Number of 5-min steps for input window (default 36=3h)")
    p.add_argument("--horizon", type=int, default=6, help="Steps ahead to predict (default 6=30min)")
    p.add_argument("--multi_step", action="store_true", help="Predict full horizon trajectory instead of last step")
    p.add_argument("--stride", type=int, default=1, help="Window stride")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=30)
    args = p.parse_args()
    main(args)
