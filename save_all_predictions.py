import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import os, joblib
import torch
import torch.nn as nn
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split

BASE = r"d:\Projects\Gridlock"
DATA = os.path.join(BASE, "dataset")
OUT  = os.path.join(BASE, "outputs")

print("Loading data...")
train_raw = pd.read_csv(os.path.join(DATA, "train.csv"))
test_raw  = pd.read_csv(os.path.join(DATA, "test.csv"))

# We don't drop index from train_raw immediately if we want to track it, but the original script drops it before split.
# To exactly match the original script's X_train and X_val:
for df in [train_raw, test_raw]:
    if 'Index' in df.columns:
        df.drop(columns=['Index'], inplace=True)

temp_med   = train_raw['Temperature'].median()
rt_mode    = train_raw['RoadType'].mode()[0]
wx_mode    = train_raw['Weather'].mode()[0]

for df in [train_raw, test_raw]:
    df['Temperature'] = df['Temperature'].fillna(temp_med)
    df['RoadType']    = df['RoadType'].fillna(rt_mode)
    df['Weather']     = df['Weather'].fillna(wx_mode)

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

num_cols = train_raw.select_dtypes(include=np.number).columns
for col in num_cols:
    q_low = train_raw[col].quantile(0.01)
    q_high = train_raw[col].quantile(0.99)
    train_raw[col] = train_raw[col].clip(lower=q_low, upper=q_high)
    if col in test_raw.columns:
        test_raw[col] = test_raw[col].clip(lower=q_low, upper=q_high)

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
    df['temp_bin'] = pd.cut(df['Temperature'], bins=5, labels=False)

TARGET  = 'demand'
CAT_OHE = ['RoadType','LargeVehicles','Landmarks','Weather','geo_p1','geo_p2','geo_p3','geo_p4']

train_raw['_split'] = 'train'
test_raw['_split']  = 'test'
combined = pd.concat([train_raw, test_raw], axis=0, ignore_index=True)
combined = pd.get_dummies(combined, columns=CAT_OHE, drop_first=False)

DROP_COLS = ['_split','timestamp','geohash']
train_enc = combined[combined['_split']=='train'].drop(columns=DROP_COLS).reset_index(drop=True)

y = train_enc[TARGET].values
X = train_enc.drop(columns=[TARGET]).values

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)

scaler = joblib.load(os.path.join(OUT, "scaler.pkl"))
X_train_sc = scaler.transform(X_train)
X_val_sc   = scaler.transform(X_val)

train_df = pd.DataFrame({'Actual_Demand': y_train})
val_df   = pd.DataFrame({'Actual_Demand': y_val})

print("Generating predictions for SKLearn models...")
# Models using scaled features
sc_models = {
    'Linear_Regression': 'linear_regression.pkl',
    'Ridge_Regression': 'ridge_regression.pkl',
    'SVR': 'svr.pkl'
}
for name, file in sc_models.items():
    model = joblib.load(os.path.join(OUT, file))
    train_df[f'{name}_Pred'] = model.predict(X_train_sc)
    val_df[f'{name}_Pred']   = model.predict(X_val_sc)

# Models using unscaled features
unsc_models = {
    'Random_Forest': 'random_forest.pkl',
    'AdaBoost': 'adaboost.pkl',
    'Gradient_Boosting': 'gradient_boosting.pkl'
}
for name, file in unsc_models.items():
    model = joblib.load(os.path.join(OUT, file))
    train_df[f'{name}_Pred'] = model.predict(X_train)
    val_df[f'{name}_Pred']   = model.predict(X_val)

print("Generating predictions for XGBoost and LightGBM...")
# XGBoost
xgb_model = xgb.XGBRegressor()
xgb_model.load_model(os.path.join(OUT, "xgboost.json"))
train_df['XGBoost_Pred'] = xgb_model.predict(X_train)
val_df['XGBoost_Pred']   = xgb_model.predict(X_val)

# LightGBM
lgb_model = lgb.Booster(model_file=os.path.join(OUT, "lightgbm.txt"))
train_df['LightGBM_Pred'] = lgb_model.predict(X_train)
val_df['LightGBM_Pred']   = lgb_model.predict(X_val)

print("Generating predictions for PyTorch Neural Networks...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_tr_t = torch.tensor(X_train_sc, dtype=torch.float32).to(device)
X_vl_t = torch.tensor(X_val_sc, dtype=torch.float32).to(device)
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

class ShallowNN(nn.Module):
    def __init__(self, n, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, 1)
        )
    def forward(self, x): return self.net(x)

bnn = BasicNN(n_feat).to(device)
bnn.load_state_dict(torch.load(os.path.join(OUT, "basic_nn_best.pt"), map_location=device))
bnn.eval()
with torch.no_grad():
    train_df['Basic_NN_Pred'] = bnn(X_tr_t).cpu().numpy().squeeze()
    val_df['Basic_NN_Pred']   = bnn(X_vl_t).cpu().numpy().squeeze()

snn = ShallowNN(n_feat, hidden=128).to(device)
snn.load_state_dict(torch.load(os.path.join(OUT, "shallow_nn_best.pt"), map_location=device))
snn.eval()
with torch.no_grad():
    train_df['Shallow_NN_Pred'] = snn(X_tr_t).cpu().numpy().squeeze()
    val_df['Shallow_NN_Pred']   = snn(X_vl_t).cpu().numpy().squeeze()

train_df.to_csv(os.path.join(BASE, "train_predictions.csv"), index=False)
val_df.to_csv(os.path.join(BASE, "val_predictions.csv"), index=False)
print("Saved train_predictions.csv and val_predictions.csv")
