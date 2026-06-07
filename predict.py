import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import os, joblib

BASE = r"d:\Projects\Gridlock"
DATA = os.path.join(BASE, "dataset")
OUT  = os.path.join(BASE, "outputs")

print("Loading data...")
train_raw = pd.read_csv(os.path.join(DATA, "train.csv"))
test_raw  = pd.read_csv(os.path.join(DATA, "test.csv"))

test_index = test_raw['Index'] if 'Index' in test_raw.columns else test_raw.index

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
test_enc  = combined[combined['_split']=='test'].drop(columns=DROP_COLS+[TARGET], errors='ignore').reset_index(drop=True)
X_test = test_enc.values

print("Loading model and predicting...")
rf_model = joblib.load(os.path.join(OUT, "random_forest.pkl"))
preds = rf_model.predict(X_test)

sub = pd.DataFrame({
    'Index': test_index,
    'demand': preds
})
sub.to_csv(os.path.join(BASE, "submission.csv"), index=False)
print("Saved to submission.csv")
