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


# NORMALIZACE
# SHOOTING
def normalize_shooting(val):
    if pd.isna(val): return False
    if str(val).strip().upper() in ['1', 'Y']: return True
    return False
df_full['SHOOTING_FLAG'] = df_full['SHOOTING'].apply(normalize_shooting)

# AVG TEMPERATURE
def categorize_tavg(t):
    if pd.isna(t):
        return 'neznámá'
    elif t < -5:
        return 'velký mráz'
    elif t <= 0:
        return 'mráz'
    elif t <= 5:
        return 'velmi chladno'
    elif t <= 10:
        return 'chladno'
    elif t <= 15:
        return 'mírně'
    elif t <= 20:
        return 'teplo'
    elif t <= 25:
        return 'velmi teplo'
    else:
        return 'horko'

df_full['TEMP_CAT'] = df_full['tavg'].apply(categorize_tavg)


# WIND - BEAUFORT
def categorize_wind_speed(w):
    if pd.isna(w):
        return 'neznámá'
    elif w < 1:
        return 'bezvětří'
    elif w <= 5:
        return 'vánek'
    elif w <= 11:
        return 'slabý vánek'
    elif w <= 19:
        return 'mírný vítr'
    elif w <= 28:
        return 'dosti čerstvý vítr'
    elif w <= 38:
        return 'čerstvý vítr'
    elif w <= 49:
        return 'silný vítr'
    elif w <= 61:
        return 'velmi silný vítr'
    elif w <= 74:
        return 'bouřlivý vítr'
    elif w <= 88:
        return 'silná vichřice'
    elif w <= 102:
        return 'vichřice'
    elif w <= 117:
        return 'prudká vichřice'
    else:
        return 'orkán'

df_full['WIND_CAT'] = df_full['wspd'].apply(categorize_wind_speed)


# PRECIPITATION
def categorize_precipitation(p):
    if pd.isna(p):
        return 'neznámá'
    elif p == 0.0:
        return 'bez srážek'
    elif p < 1.0:
        return 'velmi slabý déšť'
    elif p <= 10.0:
        return 'slabý déšť'
    elif p <= 30.0:
        return 'mírný déšť'
    elif p <= 70.0:
        return 'silný déšť'
    elif p <= 150.0:
        return 'velmi silný déšť'
    else:
        return 'extrémně silný déšť'

df_full['PRCP_CAT'] = df_full['prcp'].apply(categorize_precipitation)

# PRESSURE
def categorize_pressure_fine(p):
    if pd.isna(p):
        return 'neznámá'
    elif p < 995:
        return 'extrémně nízký tlak'
    elif p <= 999.9:
        return 'velmi nízký tlak'
    elif p <= 1005:
        return 'nízký tlak'
    elif p <= 1010:
        return 'normální tlak'
    elif p <= 1015:
        return 'vysoký tlak'
    elif p <= 1020:
        return 'velmi vysoký tlak'
    else:
        return 'extrémně vysoký tlak'

df_full['PRES_CAT'] = df_full['pres'].apply(categorize_pressure_fine)



# --- Uložení výsledku ---
full_merged_path = os.path.join(dst_dir, "filtered_merged_crime_weather.csv")
df_full.to_csv(full_merged_path, index=False)

print(f"\n✅ Spojený dataset podle společného datumového rozsahu uložen do: {full_merged_path}")

