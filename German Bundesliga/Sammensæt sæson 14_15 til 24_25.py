import pandas as pd

# Liste over dine filer
filnavne = [
    "German Bundesliga/Raw/16_17.xlsx", 
    "German Bundesliga/Raw/17_18.xlsx", "German Bundesliga/Raw/18_19.xlsx", "German Bundesliga/Raw/19_20.xlsx", 
    "German Bundesliga/Raw/20_21.xlsx", "German Bundesliga/Raw/21_22.xlsx", "German Bundesliga/Raw/22_23.xlsx", 
    "German Bundesliga/Raw/23_24.xlsx", "German Bundesliga/Raw/24_25.xlsx"
]

# Læs alle filer og sæt dem sammen
alle = pd.concat([pd.read_excel(f, engine="openpyxl") for f in filnavne],
                 ignore_index=True)

# Gem samlet fil
alle.to_excel("German Bundesliga/season_16_17_to_24_25.xlsx", index=False)

print("Færdig – samlet fil gemt")