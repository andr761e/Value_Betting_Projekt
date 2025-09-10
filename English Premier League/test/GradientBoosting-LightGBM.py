import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping
from sklearn.model_selection import train_test_split
from scipy.special import softmax
from Methods import methods
met = methods()

# Indlæs data
X = pd.read_excel("English Premier League/X.xlsx").to_numpy()
Y = pd.read_excel("English Premier League/Y.xlsx").to_numpy()
use_columns = ["Dato","FuldtidsResultat","HjemmeholdNavn","UdeholdNavn"]
match_results = pd.read_excel("English Premier League/match_results.xlsx",usecols=use_columns).to_numpy()

#Korrekt tidsafhængig split
# Definér splitpunktet baseret på antal rækker
split_point = int(0.8 * len(X))
# Tidsbaseret split
X_train, X_test = X[:split_point], X[split_point:]
Y_train, Y_test = Y[:split_point], Y[split_point:]
#Split kampresultaterne på samme måde (skal bruges senere)
match_result_split = match_results[split_point:]

# Opdel Y_train i separate arrays for hver klasse
Y_train_home = Y_train[:, 0]   # Sandsynlighed for hjemmebanesejr
Y_train_draw = Y_train[:, 1]   # Sandsynlighed for uafgjort
Y_train_away = Y_train[:, 2]   # Sandsynlighed for udebanesejr

# --- EKSTRA tidsbaseret split INDE i træning (10% til validering) ---
n_tr = int(0.9 * len(X_train))
X_tr, X_val = X_train[:n_tr], X_train[n_tr:]
Y_tr, Y_val = Y_train[:n_tr], Y_train[n_tr:]

y_tr_h, y_val_h = Y_tr[:,0], Y_val[:,0]
y_tr_d, y_val_d = Y_tr[:,1], Y_val[:,1]
y_tr_a, y_val_a = Y_tr[:,2], Y_val[:,2]

# --- MERE FLEKSIBLE men stabile LGBM-indstillinger + early stopping ---
params = dict(
    objective="l2",
    n_estimators=10000,          # stor cap + early stopping
    learning_rate=0.015,         # lav lr → præcision
    max_depth=-1,
    num_leaves=127,              # mere kapacitet
    min_child_samples=40,        # må godt splitte dybere
    feature_fraction=0.85,
    bagging_fraction=0.8,
    bagging_freq=1,
    reg_alpha=0.0,
    reg_lambda=0.5,              # lidt L2 for stabilitet
    max_bin=255,                 # bedre opløsning til kontinuerte stats
    device="gpu",
    random_state=42,
    verbose=-1
)

model_home = LGBMRegressor(**params)
model_draw = LGBMRegressor(**params)
model_away = LGBMRegressor(**params)

cb = [early_stopping(stopping_rounds=300, verbose=False)]

model_home.fit(X_tr, y_tr_h, eval_set=[(X_val, y_val_h)], callbacks=cb)
model_draw.fit(X_tr, y_tr_d, eval_set=[(X_val, y_val_d)], callbacks=cb)
model_away.fit(X_tr, y_tr_a, eval_set=[(X_val, y_val_a)], callbacks=cb)

# --- Forudsigelser ---
p_home = model_home.predict(X_test)
p_draw = model_draw.predict(X_test)
p_away = model_away.predict(X_test)

# --- Direct normalization (uden kalibrering først) ---
probs_combined = np.column_stack([p_home, p_draw, p_away])
Y_pred_proba = met.direct_normalization(probs_combined)  # din funktion

#Kalibrering
#Y_pred_proba = met.polynomial_calibration(Y_pred_proba, Y_test, degree=3)

# Manuel beregning af log-loss
epsilon = 1e-15  # For at undgå log(0)
Y_pred_proba = np.clip(Y_pred_proba, epsilon, 1 - epsilon)  # Klip sandsynlighederne
log_loss_manual = -np.mean(np.sum(Y_test * np.log(Y_pred_proba), axis=1))
print(f"Manuel Log Loss: {log_loss_manual}")

# Eksempel på sandsynlighedsfordelinger
print("Første 3 rækker af Y_test (originale sandsynlighedsfordelinger):")
print(pd.DataFrame(Y_test[:3]))

print("\nFørste 3 rækker af Y_pred_proba (forudsigede sandsynligheder):")
print(pd.DataFrame(Y_pred_proba[:3]))


#PLOTS HERFRA
np.random.seed(42)

# Plot 1: Histogram for predicted probabilities
met.plot_histogram(Y_pred_proba, classes=["Hjemmesejr", "Uafgjort", "Udesejr"], colors=["blue", "orange", "green"])

# Plot 2: Comparison of actual vs predicted probabilities (first 10 games)
met.plot_comparison(Y_pred_proba, Y_test, classes=["Hjemmesejr", "Uafgjort", "Udesejr"], colors=["blue", "orange", "green"])

# Plot 3: Correlation between actual and predicted probabilities
met.plot_correlation(Y_pred_proba, Y_test, classes=["Hjemmesejr", "Uafgjort", "Udesejr"], colors=["blue", "orange", "green"])

# Plot 4: Line chart of log-loss contributions for each sample
met.plot_log_loss(Y_pred_proba, Y_test)


#EXPORT RESULTS FOR STATISTICAL ANALYSIS
#COLUMNS TO USE

columns = [
    "Dato","FuldtidsResultat","HjemmeholdNavn","UdeholdNavn",
    "YpredH", "YpredD","YpredA", "YtrueH", "YtrueD","YtrueA","DiffH", "DiffD","DiffA",
    "OddsH","OddsD","OddsA"
]

result = pd.concat([pd.DataFrame(match_result_split), pd.concat([pd.DataFrame(Y_pred_proba), pd.concat([pd.DataFrame(Y_test),pd.concat([pd.DataFrame(Y_pred_proba - Y_test),pd.DataFrame(1/Y_test*1.03)],axis=1)], axis=1)], axis=1)],axis=1)
result.columns = columns
result.to_excel("English Premier League/test/LightGBMResultsUnfiltered.xlsx", index=False)