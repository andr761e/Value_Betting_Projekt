import pandas as pd

# Liste over dine filer
filnavne = [
    "English Premier League/Raw/14_15.xlsx", "English Premier League/Raw/15_16.xlsx", "English Premier League/Raw/16_17.xlsx", 
    "English Premier League/Raw/17_18.xlsx", "English Premier League/Raw/18_19.xlsx", "English Premier League/Raw/19_20.xlsx", 
    "English Premier League/Raw/20_21.xlsx", "English Premier League/Raw/21_22.xlsx", "English Premier League/Raw/22_23.xlsx", 
    "English Premier League/Raw/23_24.xlsx", "English Premier League/Raw/24_25.xlsx"
]

# Læs alle filer og sæt dem sammen
alle = pd.concat([pd.read_excel(f, engine="openpyxl") for f in filnavne],
                 ignore_index=True)

# Gem samlet fil
alle.to_excel("English Premier League/season_14_15_to_24_25.xlsx", index=False)

print("Færdig – samlet fil gemt")