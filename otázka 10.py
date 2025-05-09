import pandas as pd
from cleverminer import cleverminer

df = pd.read_csv('data/filtered_merged_crime_weather.csv')


crime_mapping = {
    'Aggravated Assault': 'Violent Crime',
    'Homicide': 'Violent Crime',
    'Manslaughter': 'Violent Crime',
    'Robbery': 'Violent Crime',
    'Home Invasion': 'Violent Crime',
    'Human Trafficking': 'Violent Crime',
    'Human Trafficking - Involuntary Servitude': 'Violent Crime',
    'Simple Assault': 'Violent Crime',

    'Burglary - No Property Taken': 'Property Crime',
    'Commercial Burglary': 'Property Crime',
    'Residential Burglary': 'Property Crime',
    'Other Burglary': 'Property Crime',
    'Larceny': 'Property Crime',
    'Larceny From Motor Vehicle': 'Property Crime',
    'Property Related Damage': 'Property Crime',
    'Auto Theft': 'Property Crime',
    'Auto Theft Recovery': 'Property Crime',
    'Recovered Stolen Property': 'Property Crime',
    'Arson': 'Property Crime',
    'Vandalism': 'Property Crime',

    'Drug Violation': 'Drug and Alcohol Offenses',
    'Liquor Violation': 'Drug and Alcohol Offenses',
    'Operating Under the Influence': 'Drug and Alcohol Offenses',

    'Counterfeiting': 'Fraud and Economic Crimes',
    'Embezzlement': 'Fraud and Economic Crimes',
    'Fraud': 'Fraud and Economic Crimes',
    'Confidence Games': 'Fraud and Economic Crimes',

    'Firearm Discovery': 'Weapons and Explosives',
    'Firearm Violations': 'Weapons and Explosives',
    'Ballistics': 'Weapons and Explosives',
    'Explosives': 'Weapons and Explosives',
    'Biological Threat': 'Weapons and Explosives',
    'Bomb Hoax': 'Weapons and Explosives',

    'Prostitution': 'Sexual and Moral Offenses',
    'Harassment': 'Sexual and Moral Offenses',
    'Criminal Harassment': 'Sexual and Moral Offenses',

    'Disorderly Conduct': 'Public Order and Minor Offenses',
    'Assembly or Gathering Violations': 'Public Order and Minor Offenses',
    'Phone Call Complaints': 'Public Order and Minor Offenses',
    'Verbal Disputes': 'Public Order and Minor Offenses',
    'Evading Fare': 'Public Order and Minor Offenses',
    'Landlord/Tenant Disputes': 'Public Order and Minor Offenses',
    'Gambling': 'Public Order and Minor Offenses',

    'Offenses Against Child / Family': 'Crimes Against Family and Children',
    'Restraining Order Violations': 'Crimes Against Family and Children',

    'Investigate Person': 'Investigations and Service',
    'Investigate Property': 'Investigations and Service',
    'Search Warrants': 'Investigations and Service',
    'Prisoner Related Incidents': 'Investigations and Service',
    'Police Service Incidents': 'Investigations and Service',
    'Medical Assistance': 'Investigations and Service',
    'Service': 'Investigations and Service',
    'Motor Vehicle Accident Response': 'Investigations and Service',

    'License Violation': 'Other',
    'License Plate Related Incidents': 'Other',
    'Property Found': 'Other',
    'Property Lost': 'Other',
    'Towed': 'Other',
    'Other': 'Other',
}

df['CRIME_CATEGORY'] = df['OFFENSE_CODE_GROUP'].map(crime_mapping)

df = df.dropna(subset=['CRIME_CATEGORY'])

df = df.dropna(subset=['DISTRICT'])

df = df[['CRIME_CATEGORY', 'DISTRICT', 'prcp', 'pres']]

clm = cleverminer(
    df=df, proc='SD4ftMiner',
    quantifiers={'Base1': 1000, 'Base2': 1000, 'Ratioconf': 3},
    ante={
        'attributes': [
            {'name': 'prcp', 'type': 'subset', 'minlen': 1, 'maxlen':1},
            {'name': 'pres', 'type': 'subset', 'minlen': 1, 'maxlen':1},
        ], 'minlen': 1, 'maxlen': 2, 'type': 'con'},
    succ={
        'attributes': [
            {'name': 'CRIME_CATEGORY', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
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

