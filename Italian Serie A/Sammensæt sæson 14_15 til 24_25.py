import pandas as pd

# Liste over dine filer
filnavne = [
    "Italian Serie A/Raw/16_17.xlsx", 
    "Italian Serie A/Raw/17_18.xlsx", "Italian Serie A/Raw/18_19.xlsx", "Italian Serie A/Raw/19_20.xlsx", 
    "Italian Serie A/Raw/20_21.xlsx", "Italian Serie A/Raw/21_22.xlsx", "Italian Serie A/Raw/22_23.xlsx", 
    "Italian Serie A/Raw/23_24.xlsx", "Italian Serie A/Raw/24_25.xlsx"
]

# Læs alle filer og sæt dem sammen
alle = pd.concat([pd.read_excel(f, engine="openpyxl") for f in filnavne],
                 ignore_index=True)

# Gem samlet fil
alle.to_excel("Italian Serie A/season_16_17_to_24_25.xlsx", index=False)

print("Færdig – samlet fil gemt")