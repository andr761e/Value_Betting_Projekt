import pandas as pd

# Liste over dine filer
filnavne = [
    "Dutch Eredivisie/Raw/18_19.xlsx", "Dutch Eredivisie/Raw/19_20.xlsx", 
    "Dutch Eredivisie/Raw/20_21.xlsx", "Dutch Eredivisie/Raw/21_22.xlsx", "Dutch Eredivisie/Raw/22_23.xlsx", 
    "Dutch Eredivisie/Raw/23_24.xlsx", "Dutch Eredivisie/Raw/24_25.xlsx"
]

# Læs alle filer og sæt dem sammen
alle = pd.concat([pd.read_excel(f, engine="openpyxl") for f in filnavne],
                 ignore_index=True)

# Gem samlet fil
alle.to_excel("Dutch Eredivisie/season_18_19_to_24_25.xlsx", index=False)

print("Færdig – samlet fil gemt")