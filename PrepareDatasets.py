import numpy as np
import pandas as pd
import os
from sklearn.impute import SimpleImputer
from cleverminer import *

import kagglehub
import shutil
import os


pathCrimes = kagglehub.dataset_download("jinbonnie/crime-incident-reports-in-boston")
pathWeather = kagglehub.dataset_download("swaroopmeher/boston-weather-2013-2023")

print("Path to dataset files:", pathCrimes)
print("Path to dataset files:", pathWeather)

print("Crime files:")
print(os.listdir(pathCrimes))

print("\nWeather files:")
print(os.listdir(pathWeather))

src_crime = os.path.join(pathCrimes, "12cb3883-56f5-47de-afa5-3b1cf61b257b.csv")
src_weather = os.path.join(pathWeather, "boston_weather_data.csv")

dst_dir = "data"
os.makedirs(dst_dir, exist_ok=True)

shutil.copy(src_crime, os.path.join(dst_dir, "crime.csv"))
shutil.copy(src_weather, os.path.join(dst_dir, "weather.csv"))


df_crime = pd.read_csv("data/crime.csv")
df_weather = pd.read_csv("data/weather.csv")

print("=== Crime data ===")
print(df_crime.shape)
print(df_crime.columns)
print(df_crime.head(3))
print(df_crime.info())
print(df_crime.isnull().sum())

print("\n=== Weather data ===")
print(df_weather.shape)
print(df_weather.columns)
print(df_weather.head(3))
print(df_weather.info())
print(df_weather.isnull().sum())


# --- Převedení na datetime a extrakce datumu ---
df_crime['OCCURRED_ON_DATE'] = pd.to_datetime(df_crime['OCCURRED_ON_DATE'], errors='coerce')
df_crime['DATE'] = df_crime['OCCURRED_ON_DATE'].dt.date

df_weather['time'] = pd.to_datetime(df_weather['time'], errors='coerce')
df_weather['DATE'] = df_weather['time'].dt.date

# --- Nejnižší a nejvyšší datum v každém datasetu ---
crime_min = df_crime['DATE'].min()
crime_max = df_crime['DATE'].max()

weather_min = df_weather['DATE'].min()
weather_max = df_weather['DATE'].max()

# --- Výpis rozsahů ---
print(f"\n📅 Datumový rozsah CRIME:   {crime_min} až {crime_max}")
print(f"📅 Datumový rozsah WEATHER: {weather_min} až {weather_max}")

# --- Společný rozsah (průnik) ---
common_start = max(crime_min, weather_min)
common_end = min(crime_max, weather_max)

print(f"\n🔗 SPOLEČNÝ ROZSAH: {common_start} až {common_end}")

# --- Filtrování podle průniku ---
df_crime_filtered = df_crime[(df_crime['DATE'] >= common_start) & (df_crime['DATE'] <= common_end)]
df_weather_filtered = df_weather[(df_weather['DATE'] >= common_start) & (df_weather['DATE'] <= common_end)]

print(f"\n📊 Počet záznamů po filtrování:")
print(f"  CRIME:   {df_crime_filtered.shape[0]}")
print(f"  WEATHER: {df_weather_filtered.shape[0]}")

# MISSING VALUES FILL
df_weather_filtered['time'] = pd.to_datetime(df_weather_filtered['time'], errors='coerce')

def fill_missing_value_no_cols(df, column):
    df_copy = df.copy()

    for i in df_copy.index:
        if pd.isna(df_copy.loc[i, column]):
            date = df_copy.loc[i, 'time']
            if pd.isna(date):
                continue

            year = date.year
            month = date.month

            nearby = df_copy[
                (df_copy['time'].dt.year == year) &
                (df_copy['time'].dt.month.isin([month - 1, month, month + 1])) &
                (~df_copy[column].isna())
            ][column]

            all_years = df_copy[
                (df_copy['time'].dt.month == month) &
                (~df_copy[column].isna())
            ][column]

            combined = pd.concat([nearby, all_years])
            if not combined.empty:
                df_copy.at[i, column] = combined.mean()

    return df_copy[column]


print("\n🔧 Imputace tavg bez vytváření sloupců...")
df_weather_filtered['tavg'] = fill_missing_value_no_cols(df_weather_filtered, 'tavg')
print("✅ Hotovo. Zbývající NaN v tavg:", df_weather_filtered['tavg'].isna().sum())

print("\n🔧 Imputace pres bez vytváření sloupců...")
df_weather_filtered['pres'] = fill_missing_value_no_cols(df_weather_filtered, 'pres')
print("✅ Hotovo. Zbývající NaN v pres:", df_weather_filtered['pres'].isna().sum())


df_full = pd.merge(df_crime_filtered, df_weather_filtered, on='DATE', how='inner')


# SHOOTING → přepis na boolean
def normalize_shooting(val):
    if pd.isna(val): return False
    return str(val).strip().upper() in ['1', 'Y']

df_full['SHOOTING'] = df_full['SHOOTING'].apply(normalize_shooting)

# AVG TEMPERATURE (tavg → kategorie)
def categorize_tavg(t):
    if pd.isna(t):
        return 'unknown'
    elif t < -5:
        return 'hard freezing'
    elif t <= 0:
        return 'freezing'
    elif t <= 5:
        return 'very cold'
    elif t <= 10:
        return 'cold'
    elif t <= 15:
        return 'fresh'
    elif t <= 20:
        return 'warm'
    elif t <= 25:
        return 'very warm'
    else:
        return 'hot'

df_full['tavg'] = df_full['tavg'].apply(categorize_tavg)

# WIND SPEED (wspd → kategorie)
def categorize_wind_speed(w):
    if pd.isna(w):
        return 'unknown'
    elif w < 1:
        return 'Calm'
    elif w <= 5:
        return 'Light air'
    elif w <= 11:
        return 'Light breeze'
    elif w <= 19:
        return 'Gentle breeze'
    elif w <= 28:
        return 'Moderate breeze'
    elif w <= 38:
        return 'Fresh breeze'
    elif w <= 49:
        return 'Strong breeze'
    elif w <= 61:
        return 'Moderate gale'
    elif w <= 74:
        return 'Fresh gale'
    elif w <= 88:
        return 'Severe gale'
    elif w <= 102:
        return 'Storm'
    elif w <= 117:
        return 'Violent storm'
    else:
        return 'Hurricane'

df_full['wspd'] = df_full['wspd'].apply(categorize_wind_speed)

# PRECIPITATION (prcp → kategorie)
def categorize_precipitation(p):
    if pd.isna(p):
        return 'unknown'
    elif p == 0.0:
        return 'no rain'
    elif p < 1.0:
        return 'very light'
    elif p <= 10.0:
        return 'light'
    elif p <= 30.0:
        return 'medium'
    elif p <= 70.0:
        return 'strong'
    elif p <= 150.0:
        return 'very strong'
    else:
        return 'extremely strong'

df_full['prcp'] = df_full['prcp'].apply(categorize_precipitation)

# PRESSURE (pres → kategorie)
def categorize_pressure_fine(p):
    if pd.isna(p):
        return 'unknown'
    elif p < 995:
        return 'extremely low'
    elif p <= 999.9:
        return 'very low'
    elif p <= 1005:
        return 'low'
    elif p <= 1010:
        return 'normal'
    elif p <= 1015:
        return 'high'
    elif p <= 1020:
        return 'very high'
    else:
        return 'extremely high'

df_full['pres'] = df_full['pres'].apply(categorize_pressure_fine)




# --- Uložení výsledku ---
full_merged_path = os.path.join(dst_dir, "filtered_merged_crime_weather.csv")
df_full.to_csv(full_merged_path, index=False)

print(f"\n✅ Spojený dataset podle společného datumového rozsahu uložen do: {full_merged_path}")

