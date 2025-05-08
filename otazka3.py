import pandas as pd
from cleverminer import cleverminer

df = pd.read_csv('data/filtered_merged_crime_weather.csv')


clm = cleverminer(
    df=df,
    target='OFFENSE_CODE_GROUP',
    proc='CFMiner',
    quantifiers={'S_Up': 1, 'S_Down': 3, 'Base': 500},
    cond={
        'attributes': [
            {'name': 'wspd', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
            {'name': 'SHOOTING', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
            {'name': 'DAY_OF_WEEK', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
        ],
        'minlen': 2,
        'maxlen': 3,
        'type': 'con'
    }
)

clm.print_summary()
clm.print_rulelist()
clm.print_rule(3)