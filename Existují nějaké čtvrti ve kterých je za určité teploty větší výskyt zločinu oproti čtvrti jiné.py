import pandas as pd
from cleverminer import cleverminer

# Načtení tvého datasetu
df = pd.read_csv('data/filtered_merged_crime_weather.csv')

mapping_tavg = {
    'hard freezing':  1,
    'freezing':       2,
    'very cold':      3,
    'cold':           4,
    'fresh':          5,
    'warm':           6,
    'very warm':      7,
    'hot':            8
}
df['tavg_code'] = df['tavg'].map(mapping_tavg)

df = df.dropna(subset=['DISTRICT'])

df = df[['OFFENSE_CODE_GROUP', 'DISTRICT', 'tavg_code']]

clm = cleverminer(
    df=df, proc='SD4ftMiner',
    quantifiers={'Base1': 100, 'Base2': 1000, 'Ratioconf': 3},
    ante={
        'attributes': [
            {'name': 'tavg_code', 'type': 'seq', 'minlen': 1, 'maxlen':2},
        ], 'minlen': 1, 'maxlen': 2, 'type': 'con'},
    succ={
        'attributes': [
            {'name': 'OFFENSE_CODE_GROUP', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
        ], 'minlen': 1, 'maxlen': 1, 'type': 'con'},
    frst={
        'attributes': [
            {'name': 'DISTRICT', 'type': 'subset', 'minlen': 1, 'maxlen': 1}
        ], 'minlen': 1, 'maxlen': 1, 'type': 'con'},
    scnd={
        'attributes': [
            {'name': 'DISTRICT', 'type': 'subset', 'minlen': 1, 'maxlen': 1}
        ], 'minlen': 1, 'maxlen': 1, 'type': 'con'}
)

clm.print_summary()
clm.print_rulelist()

