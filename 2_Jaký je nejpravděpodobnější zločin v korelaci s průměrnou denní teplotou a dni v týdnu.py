import pandas as pd
from cleverminer import cleverminer

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

# mapping_wspd = {
#     'calm':             1,  # < 1 m/s
#     'light air':        2,  # 1–5 m/s
#     'light breeze':     3,  # 5–11 m/s
#     'gentle breeze':    4,  # 11–19 m/s
#     'moderate breeze':  5,  # 19–28 m/s
#     'fresh breeze':     6,  # 28–38 m/s
#     'strong breeze':    7,  # 38–49 m/s
#     'moderate gale':    8,  # 49–61 m/s
#     'fresh gale':       9,  # 61–74 m/s
#     'severe gale':     10,  # 74–88 m/s
#     'storm':           11,  # 88–102 m/s
#     'violent storm':   12,  # 102–117 m/s
#     'hurricane':       13   # > 117 m/s
# }
#
#
# df['wspd_code'] = df['wspd'].map(mapping_wspd)
df = df.dropna(subset=['tavg_code', 'DAY_OF_WEEK', 'OFFENSE_CODE_GROUP'])
df = df[['tavg_code', 'DAY_OF_WEEK', 'OFFENSE_CODE_GROUP']]

clm = cleverminer(
    df=df,
    proc='4ftMiner',
    quantifiers={'conf': 0.2, 'Base': 4000},
    ante={
        'attributes': [
            {'name': 'tavg_code', 'type': 'seq', 'minlen': 1, 'maxlen': 2},
            {'name': 'DAY_OF_WEEK', 'type': 'subset', 'minlen': 1, 'maxlen': 1}
        ],
        'type': 'con', 'minlen': 2, 'maxlen': 3
    },
    succ={
        'attributes': [
            {'name': 'OFFENSE_CODE_GROUP', 'type': 'subset', 'minlen': 1, 'maxlen': 2}
        ],
        'type': 'con', 'minlen': 1, 'maxlen': 2
    }
)

clm.print_summary()
clm.print_rulelist()
