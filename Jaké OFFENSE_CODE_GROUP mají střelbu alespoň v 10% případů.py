import pandas as pd
from cleverminer import cleverminer
from io import StringIO
from contextlib import redirect_stdout

df = pd.read_csv('data/filtered_merged_crime_weather.csv')
df = df[['OFFENSE_CODE_GROUP', 'SHOOTING']].dropna(subset=['OFFENSE_CODE_GROUP']).copy()

clm = cleverminer(
    df=df,
    proc='4ftMiner',
    quantifiers={
        'conf': 0.1,
        'imp': 0.0,
        'support': 0.0
    },
    ante={
        'attributes': [
            {'name': 'OFFENSE_CODE_GROUP', 'type': 'subset', 'minlen': 1, 'maxlen': 1}
        ],
        'type': 'con',
        'minlen': 1,
        'maxlen': 1
    },
    succ={
        'attributes': [
            {'name': 'SHOOTING', 'type': 'subset', 'minlen': 1, 'maxlen': 1}
        ],
        'type': 'con',
        'minlen': 1,
        'maxlen': 1
    }
)

clm.print_summary()

buf = StringIO()
with redirect_stdout(buf):
    clm.print_rulelist()

text = buf.getvalue()

for line in text.splitlines():
    if 'SHOOTING(True)' in line:
        print(line)
