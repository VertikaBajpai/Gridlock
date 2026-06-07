import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os, joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import xgboost as xgb
import lightgbm as lgb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------
# 0. Paths
# ---------------------------------------------
BASE = r"d:\Projects\Gridlock"
DATA = os.path.join(BASE, "dataset")
OUT  = os.path.join(BASE, "outputs")
os.makedirs(OUT, exist_ok=True)

print("=" * 60)
print("DEMAND PREDICTION PIPELINE")
print("=" * 60)

# ---------------------------------------------
# 1. Load
# ---------------------------------------------
print("\n[1] Loading data...")
train_raw = pd.read_csv(os.path.join(DATA, "train.csv"))
test_raw  = pd.read_csv(os.path.join(DATA, "test.csv"))
print(f"  Train: {train_raw.shape}  |  Test: {test_raw.shape}")
print("\n  Train dtypes:\n", train_raw.dtypes.to_string())
print("\n  Sample rows:")
print(train_raw.head(3).to_string())

# ---------------------------------------------
# 2. Drop index
# ---------------------------------------------
for df in [train_raw, test_raw]:
    if 'Index' in df.columns:
        df.drop(columns=['Index'], inplace=True)
print("\n[2] Index column dropped.")

# ---------------------------------------------
# 3. Null analysis + imputation
# ---------------------------------------------
print("\n[3] Null value analysis...")
print("\n  Train nulls:\n", train_raw.isnull().sum().to_string())
print("\n  Test  nulls:\n", test_raw.isnull().sum().to_string())

# Fit on train, apply to both
temp_med   = train_raw['Temperature'].median()
rt_mode    = train_raw['RoadType'].mode()[0]
wx_mode    = train_raw['Weather'].mode()[0]

for df in [train_raw, test_raw]:
    df['Temperature'] = df['Temperature'].fillna(temp_med)
    df['RoadType']    = df['RoadType'].fillna(rt_mode)
    df['Weather']     = df['Weather'].fillna(wx_mode)

print("\n  After imputation - Train nulls:\n", train_raw.isnull().sum().to_string())
print("\n  Temperature median used:", round(temp_med, 3))
print("  RoadType mode used     :", rt_mode)
print("  Weather mode used      :", wx_mode)

# ---------------------------------------------
# 4. Geohash decode helper
# ---------------------------------------------
def geohash_decode(gh):
    BASE32 = '0123456789bcdefghjkmnpqrstuvwxyz'
    lat_iv = [-90.0, 90.0]
    lon_iv = [-180.0, 180.0]
    is_lon = True
    for c in gh:
        cd = BASE32.index(c)
        for bits in [16, 8, 4, 2, 1]:
            if is_lon:
                mid = sum(lon_iv) / 2
                if cd & bits: lon_iv[0] = mid
                else:          lon_iv[1] = mid
            else:
                mid = sum(lat_iv) / 2
                if cd & bits: lat_iv[0] = mid
                else:          lat_iv[1] = mid
            is_lon = not is_lon
    return sum(lat_iv)/2, sum(lon_iv)/2

def add_geohash_features(df):
    coords = df['geohash'].apply(geohash_decode)
    df['geo_lat'] = coords.apply(lambda x: x[0])
    df['geo_lon'] = coords.apply(lambda x: x[1])
    df['geo_p1']  = df['geohash'].str[:1]
    df['geo_p2']  = df['geohash'].str[:2]
    df['geo_p3']  = df['geohash'].str[:3]
    df['geo_p4']  = df['geohash'].str[:4]
    return df

def parse_timestamp(df):
    df['hour']   = df['timestamp'].apply(lambda x: int(str(x).split(':')[0]))
    df['minute'] = df['timestamp'].apply(lambda x: int(str(x).split(':')[1]))
    df['time_of_day_min'] = df['hour']*60 + df['minute']
    df['hour_sin']   = np.sin(2*np.pi*df['hour']/24)
    df['hour_cos']   = np.cos(2*np.pi*df['hour']/24)
    df['minute_sin'] = np.sin(2*np.pi*df['minute']/60)
    df['minute_cos'] = np.cos(2*np.pi*df['minute']/60)
    return df

# ---------------------------------------------
# 5. Feature Engineering
# ---------------------------------------------
print("\n[4] Feature Engineering...")
train_raw = add_geohash_features(train_raw)
test_raw  = add_geohash_features(test_raw)

train_raw = parse_timestamp(train_raw)
test_raw  = parse_timestamp(test_raw)

for df in [train_raw, test_raw]:
    df['day_sin'] = np.sin(2*np.pi*df['day']/7)
    df['day_cos'] = np.cos(2*np.pi*df['day']/7)
    df['lanes_x_vehicles'] = df['NumberofLanes'] * (df['LargeVehicles']=='Allowed').astype(int)
    df['is_peak']  = ((df['hour'].between(7,9)) | (df['hour'].between(17,19))).astype(int)
    df['is_night'] = ((df['hour']>=22) | (df['hour']<=5)).astype(int)
    # Temp bins
    df['temp_bin'] = pd.cut(df['Temperature'], bins=5, labels=False)

print("  Geohash lat/lon, prefix levels (p1-p4)")
print("  Cyclical encoding: hour, minute, day")
print("  Flags: is_peak, is_night")
print("  Interaction: lanes x vehicles")
print("  Temp bins")

# ---------------------------------------------
# 6. One-Hot Encoding
# ---------------------------------------------
print("\n[5] One-Hot Encoding...")
TARGET  = 'demand'
CAT_OHE = ['RoadType','LargeVehicles','Landmarks','Weather','geo_p1','geo_p2','geo_p3','geo_p4']

train_raw['_split'] = 'train'
test_raw['_split']  = 'test'
combined = pd.concat([train_raw, test_raw], axis=0, ignore_index=True)
combined = pd.get_dummies(combined, columns=CAT_OHE, drop_first=False)

DROP_COLS = ['_split','timestamp','geohash']
train_enc = combined[combined['_split_x'] if '_split_x' in combined.columns else combined['_split']=='train'].copy() if False else combined[combined['_split']=='train'].drop(columns=DROP_COLS).reset_index(drop=True)
test_enc  = combined[combined['_split']=='test'].drop(columns=DROP_COLS+[TARGET]).reset_index(drop=True)

print(f"  Encoded train: {train_enc.shape}")
print(f"  Encoded test : {test_enc.shape}")
print(f"  Feature columns: {list(train_enc.columns[:10])} ...")

# ---------------------------------------------
# 7. EDA
# ---------------------------------------------
print("\n[6] EDA...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Exploratory Data Analysis - Demand Prediction", fontsize=15, fontweight='bold')

# Demand distribution
axes[0,0].hist(train_enc[TARGET], bins=60, color='steelblue', edgecolor='white', alpha=0.85)
axes[0,0].set_title("Demand Distribution")
axes[0,0].set_xlabel("Demand"); axes[0,0].set_ylabel("Count")
axes[0,0].axvline(train_enc[TARGET].mean(), color='red', linestyle='--', label=f"Mean={train_enc[TARGET].mean():.3f}")
axes[0,0].legend()

# Demand by hour
hd = train_enc.groupby('hour')[TARGET].mean()
axes[0,1].plot(hd.index, hd.values, marker='o', color='coral', linewidth=2)
axes[0,1].fill_between(hd.index, hd.values, alpha=0.15, color='coral')
axes[0,1].set_title("Avg Demand by Hour")
axes[0,1].set_xlabel("Hour"); axes[0,1].set_ylabel("Mean Demand")

# Demand by day
dd = train_enc.groupby('day')[TARGET].mean()
axes[0,2].bar(dd.index, dd.values, color='mediumpurple', alpha=0.85)
axes[0,2].set_title("Avg Demand by Day")
axes[0,2].set_xlabel("Day"); axes[0,2].set_ylabel("Mean Demand")

# Temperature vs Demand
axes[1,0].scatter(train_enc['Temperature'], train_enc[TARGET], alpha=0.15, s=4, color='teal')
axes[1,0].set_title("Temperature vs Demand")
axes[1,0].set_xlabel("Temperature"); axes[1,0].set_ylabel("Demand")

# Lanes vs Demand
lane_groups = [train_enc[train_enc['NumberofLanes']==n][TARGET].values
               for n in sorted(train_enc['NumberofLanes'].unique())]
axes[1,1].boxplot(lane_groups, labels=[str(n) for n in sorted(train_enc['NumberofLanes'].unique())])
axes[1,1].set_title("Demand by NumberOfLanes")
axes[1,1].set_xlabel("Lanes"); axes[1,1].set_ylabel("Demand")

# Geo-spatial
sc = axes[1,2].scatter(train_enc['geo_lon'], train_enc['geo_lat'],
                       c=train_enc[TARGET], cmap='YlOrRd', alpha=0.25, s=4)
plt.colorbar(sc, ax=axes[1,2], label='Demand')
axes[1,2].set_title("Geo-spatial Demand Heatmap")
axes[1,2].set_xlabel("Longitude"); axes[1,2].set_ylabel("Latitude")

plt.tight_layout()
plt.savefig(os.path.join(OUT, "eda.png"), dpi=120, bbox_inches='tight')
plt.close()
print("  EDA plot saved -> outputs/eda.png")

# Correlation with demand
num_cols = train_enc.select_dtypes(include=np.number).columns.tolist()
corr_series = train_enc[num_cols].corr()[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
top20 = corr_series.head(20).to_frame()

plt.figure(figsize=(8, 8))
colors = ['green' if v>0 else 'crimson' for v in top20[TARGET]]
top20[TARGET].plot(kind='barh', color=colors, alpha=0.8)
plt.title("Top 20 Feature Correlations with Demand")
plt.xlabel("Pearson Correlation")
plt.axvline(0, color='black', linewidth=0.8)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "correlation.png"), dpi=120, bbox_inches='tight')
plt.close()
print("  Correlation plot saved -> outputs/correlation.png")

print("\n  Basic stats:")
print(train_enc[[TARGET,'Temperature','NumberofLanes','hour','day']].describe().to_string())

# ---------------------------------------------
# 8. Train/Val split (85/15)
# ---------------------------------------------
print("\n[7] Train/Val split (85/15)...")
y = train_enc[TARGET].values
X = train_enc.drop(columns=[TARGET]).values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)
X_test = test_enc.values
print(f"  X_train: {X_train.shape} | X_val: {X_val.shape} | X_test: {X_test.shape}")

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc   = scaler.transform(X_val)
X_test_sc  = scaler.transform(X_test)
joblib.dump(scaler, os.path.join(OUT, "scaler.pkl"))
print("  Scaler saved -> outputs/scaler.pkl")

# ---------------------------------------------
# 9. Metrics helper
# ---------------------------------------------
results = []

def evaluate(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100
    print(f"\n  [{name}]")
    print(f"    MAE  : {mae:.5f}")
    print(f"    RMSE : {rmse:.5f}")
    print(f"    R2   : {r2:.5f}")
    print(f"    MAPE : {mape:.2f}%")
    results.append({'Model': name, 'MAE': mae, 'RMSE': rmse, 'R2': r2, 'MAPE(%)': mape})
    return y_pred

# ---------------------------------------------
# 10. ML Models
# ---------------------------------------------
print("\n[8] Training ML Models...")
print("-" * 50)

# Linear Regression
print("\n  Training: Linear Regression")
lr = LinearRegression()
lr.fit(X_train_sc, y_train)
evaluate("Linear Regression", y_val, lr.predict(X_val_sc))
joblib.dump(lr, os.path.join(OUT, "linear_regression.pkl"))

# Ridge Regression
print("\n  Training: Ridge Regression")
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_sc, y_train)
evaluate("Ridge Regression", y_val, ridge.predict(X_val_sc))
joblib.dump(ridge, os.path.join(OUT, "ridge_regression.pkl"))

# SVR (on subset for speed)
print("\n  Training: SVR (subset of 10k samples for speed)")
svr = SVR(kernel='rbf', C=10, epsilon=0.01)
svr.fit(X_train_sc[:10000], y_train[:10000])
evaluate("SVR (RBF)", y_val, svr.predict(X_val_sc))
joblib.dump(svr, os.path.join(OUT, "svr.pkl"))

# XGBoost
print("\n  Training: XGBoost")
xgb_model = xgb.XGBRegressor(
    n_estimators=500, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1, verbosity=0
)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
evaluate("XGBoost", y_val, xgb_model.predict(X_val))
xgb_model.save_model(os.path.join(OUT, "xgboost.json"))

# LightGBM
print("\n  Training: LightGBM")
lgb_model = lgb.LGBMRegressor(
    n_estimators=500, learning_rate=0.05, num_leaves=63,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1, verbose=-1
)
lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)]
)
evaluate("LightGBM", y_val, lgb_model.predict(X_val))
lgb_model.booster_.save_model(os.path.join(OUT, "lightgbm.txt"))

# ---------------------------------------------
# 11. Basic Neural Network
# ---------------------------------------------
print("\n[9] Basic Neural Network (PyTorch)...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Device: {device}")

def to_tensors(X, y=None):
    Xt = torch.tensor(X, dtype=torch.float32).to(device)
    if y is not None:
        return Xt, torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device)
    return Xt

X_tr_t, y_tr_t = to_tensors(X_train_sc, y_train)
X_vl_t, y_vl_t = to_tensors(X_val_sc,   y_val)
tr_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=512, shuffle=True)

n_feat = X_train_sc.shape[1]

class BasicNN(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.2),
            nn.Linear(128, 64),  nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x): return self.net(x)

bnn = BasicNN(n_feat).to(device)
opt = torch.optim.Adam(bnn.parameters(), lr=1e-3, weight_decay=1e-4)
sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
crit = nn.MSELoss()

EPOCHS = 50
bnn_tr_loss, bnn_vl_loss = [], []
best_bnn = float('inf')

for ep in range(1, EPOCHS+1):
    bnn.train()
    ep_loss = 0
    for Xb, yb in tr_loader:
        opt.zero_grad()
        loss = crit(bnn(Xb), yb)
        loss.backward(); opt.step()
        ep_loss += loss.item() * len(Xb)
    ep_loss /= len(X_tr_t)

    bnn.eval()
    with torch.no_grad():
        vl = crit(bnn(X_vl_t), y_vl_t).item()
    sch.step(vl)
    bnn_tr_loss.append(ep_loss); bnn_vl_loss.append(vl)

    if vl < best_bnn:
        best_bnn = vl
        torch.save(bnn.state_dict(), os.path.join(OUT, "basic_nn_best.pt"))

    if ep % 10 == 0:
        print(f"    Epoch {ep:3d}/{EPOCHS} | Train MSE: {ep_loss:.5f} | Val MSE: {vl:.5f}")

bnn.load_state_dict(torch.load(os.path.join(OUT, "basic_nn_best.pt"), map_location=device, weights_only=True))
bnn.eval()
with torch.no_grad():
    bnn_preds = bnn(X_vl_t).cpu().numpy().squeeze()
evaluate("Basic Neural Network", y_val, bnn_preds)

plt.figure(figsize=(8, 4))
plt.plot(bnn_tr_loss, label='Train MSE')
plt.plot(bnn_vl_loss, label='Val MSE')
plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.title("Basic NN Training Curves")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT, "basic_nn_curves.png"), dpi=120)
plt.close()
print("  Basic NN weights saved -> outputs/basic_nn_best.pt")

# ---------------------------------------------
# 12. Shallow Neural Network
# ---------------------------------------------
print("\n[10] Shallow Neural Network (1 hidden layer)...")

class ShallowNN(nn.Module):
    def __init__(self, n, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, 1)
        )
    def forward(self, x): return self.net(x)

snn = ShallowNN(n_feat, hidden=128).to(device)
opt_s = torch.optim.Adam(snn.parameters(), lr=5e-4, weight_decay=1e-4)
sch_s = torch.optim.lr_scheduler.CosineAnnealingLR(opt_s, T_max=80)

EPOCHS_S = 80
snn_tr_loss, snn_vl_loss = [], []
best_snn = float('inf')

for ep in range(1, EPOCHS_S+1):
    snn.train()
    ep_loss = 0
    for Xb, yb in tr_loader:
        opt_s.zero_grad()
        loss = crit(snn(Xb), yb)
        loss.backward(); opt_s.step()
        ep_loss += loss.item() * len(Xb)
    ep_loss /= len(X_tr_t)
    sch_s.step()

    snn.eval()
    with torch.no_grad():
        vl = crit(snn(X_vl_t), y_vl_t).item()
    snn_tr_loss.append(ep_loss); snn_vl_loss.append(vl)

    if vl < best_snn:
        best_snn = vl
        torch.save(snn.state_dict(), os.path.join(OUT, "shallow_nn_best.pt"))

    if ep % 10 == 0:
        print(f"    Epoch {ep:3d}/{EPOCHS_S} | Train MSE: {ep_loss:.5f} | Val MSE: {vl:.5f}")

snn.load_state_dict(torch.load(os.path.join(OUT, "shallow_nn_best.pt"), map_location=device, weights_only=True))
snn.eval()
with torch.no_grad():
    snn_preds = snn(X_vl_t).cpu().numpy().squeeze()
evaluate("Shallow Neural Network", y_val, snn_preds)

plt.figure(figsize=(8, 4))
plt.plot(snn_tr_loss, label='Train MSE')
plt.plot(snn_vl_loss, label='Val MSE')
plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.title("Shallow NN Training Curves")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT, "shallow_nn_curves.png"), dpi=120)
plt.close()
print("  Shallow NN weights saved -> outputs/shallow_nn_best.pt")

# ---------------------------------------------
# 13. Results summary
# ---------------------------------------------
print("\n" + "="*60)
print("MODEL COMPARISON SUMMARY (Validation Set)")
print("="*60)
results_df = pd.DataFrame(results).sort_values('RMSE')
print(results_df.to_string(index=False))
results_df.to_csv(os.path.join(OUT, "model_results.csv"), index=False)

# Bar chart comparison
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Model Comparison - Validation Set", fontsize=13, fontweight='bold')
colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(results_df)))

for ax, metric in zip(axes, ['MAE','RMSE','R2']):
    bars = ax.barh(results_df['Model'], results_df[metric], color=colors)
    ax.set_xlabel(metric); ax.set_title(f"Val {metric}"); ax.invert_yaxis()
    for bar, val in zip(bars, results_df[metric]):
        ax.text(bar.get_width()*1.01, bar.get_y()+bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUT, "model_comparison.png"), dpi=120, bbox_inches='tight')
plt.close()

# XGBoost feature importance
feat_names = train_enc.drop(columns=[TARGET]).columns.tolist()
xgb_imp = pd.Series(xgb_model.feature_importances_, index=feat_names).sort_values(ascending=False)

plt.figure(figsize=(10, 7))
xgb_imp.head(20).plot(kind='barh', color='steelblue', alpha=0.85)
plt.gca().invert_yaxis()
plt.title("Top 20 Feature Importances (XGBoost)")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "feature_importance.png"), dpi=120)
plt.close()

# Neural network curves combined
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(bnn_tr_loss, label='Train'); axes[0].plot(bnn_vl_loss, label='Val')
axes[0].set_title("Basic NN Curves"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MSE"); axes[0].legend()
axes[1].plot(snn_tr_loss, label='Train'); axes[1].plot(snn_vl_loss, label='Val')
axes[1].set_title("Shallow NN Curves"); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("MSE"); axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "nn_training_curves.png"), dpi=120)
plt.close()

print("\n[DONE] All outputs saved to:", OUT)
print("  Plots   : eda.png, correlation.png, model_comparison.png, feature_importance.png, nn_training_curves.png")
print("  Weights : basic_nn_best.pt, shallow_nn_best.pt")
print("  Models  : xgboost.json, lightgbm.txt, linear_regression.pkl, ridge_regression.pkl, svr.pkl")
print("  Scaler  : scaler.pkl")
print("  Results : model_results.csv")
