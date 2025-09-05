import pandas as pd

# Liste over dine filer
filnavne = [
    "Spanish La Liga/Raw/16_17.xlsx", 
    "Spanish La Liga/Raw/17_18.xlsx", "Spanish La Liga/Raw/18_19.xlsx", "Spanish La Liga/Raw/19_20.xlsx", 
    "Spanish La Liga/Raw/20_21.xlsx", "Spanish La Liga/Raw/21_22.xlsx", "Spanish La Liga/Raw/22_23.xlsx", 
    "Spanish La Liga/Raw/23_24.xlsx", "Spanish La Liga/Raw/24_25.xlsx"
]

# Læs alle filer og sæt dem sammen
alle = pd.concat([pd.read_excel(f, engine="openpyxl") for f in filnavne],
                 ignore_index=True)

# Gem samlet fil
alle.to_excel("Spanish La Liga/season_16_17_to_24_25.xlsx", index=False)

print("Færdig – samlet fil gemt")