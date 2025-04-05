import pandas as pd

# Načti dataset
df = pd.read_csv("data/filtered_merged_crime_weather.csv")

# Nastavení pro konzolový výstup
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 200)

# Pomocná funkce pro ignorované sloupce
def is_excluded(col):
    col_lower = col.lower()
    return any(x in col_lower for x in ['date', 'time', 'lat', 'long'])

# Přesměrování výstupu do souboru
with open("profil_datasets.txt", "w", encoding="utf-8") as f:
    def write(line=""):
        f.write(line + "\n")

    print("=== Přehled datových typů ===")
    write("=== Přehled datových typů ===")
    type_summary = df.dtypes.value_counts()
    print(type_summary)
    write(str(type_summary))
    write()

    # Kategorie proměnných
    categorical = df.select_dtypes(include='object')
    print("=== Kategorie proměnných (zkráceně) ===")
    write("=== Kategorie proměnných ===")
    for col in categorical.columns:
        if is_excluded(col):
            continue

        unique_vals = df[col].dropna().unique()
        print(f"{col:30} | {len(unique_vals):5} hodnot")
        write(f"{col:30} | unikátních: {len(unique_vals):5} | chybějících: {df[col].isnull().sum():5}")
        for val in unique_vals:
            write(f"  - {val}")
        write()

    # Číselné proměnné
    numerical = df.select_dtypes(include=['int64', 'float64'])
    print("\n=== Statistika číselných proměnných ===")
    stats = numerical.describe().T
    print(stats.to_string())
    write("=== Číselné proměnné ===")
    write(stats.to_string())
    write()

    # Chybějící hodnoty
    write("=== Chybějící hodnoty v číselných sloupcích ===")
    print("\n=== Chybějící hodnoty v číselných sloupcích ===")
    for col in numerical.columns:
        missing = df[col].isnull().sum()
        if missing > 0:
            print(f"{col:30} | chybějících: {missing}")
            write(f"{col:30} | chybějících: {missing}")
