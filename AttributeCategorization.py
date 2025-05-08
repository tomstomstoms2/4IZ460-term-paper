import pandas as pd

# Načti dataset
df = pd.read_csv("data/filtered_merged_crime_weather.csv")

# Nastavení pro konzolový výstup
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 200)

# Pomocná funkce pro ignorované sloupce
def is_excluded(col):
    col_lower = col.lower()
    return any(x in col_lower for x in ['date', 'time', 'lat', 'long', 'location', 'Incident'])

# Přesměrování výstupu do souboru
with open("data/profil_datasets.txt", "w", encoding="utf-8") as f:
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
        unique_vals = df[col].dropna().unique()
        num_unique = len(unique_vals)
        num_missing = df[col].isnull().sum()

        # Vždy vypiš shrnutí
        print(f"{col:30} | {num_unique:5} hodnot")
        write(f"{col:30} | unikátních: {num_unique:5} | chybějících: {num_missing:5}")

        # Hodnoty vypiš pouze pokud sloupec není ignorován
        if not is_excluded(col):
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
    for col in categorical.columns:
        unique_vals = df[col].dropna().unique()
        num_unique = len(unique_vals)
        num_missing = df[col].isnull().sum()

        # Vždy vypiš shrnutí
        print(f"{col:30} | {num_unique:5} hodnot")
        write(f"{col:30} | unikátních: {num_unique:5} | chybějících: {num_missing:5}")

        # Hodnoty vypiš pouze pokud sloupec není ignorován
        if not is_excluded(col):
            for val in unique_vals:
                write(f"  - {val}")
        write()

