import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from scipy.special import softmax
from Methods import methods
met = methods()
import os
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks

# Sæt seeds for fuld reproducerbarhed
os.environ['PYTHONHASHSEED'] = str(42)
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# (valgfrit, men godt): Tving determinisme i TensorFlow
os.environ['TF_DETERMINISTIC_OPS'] = '1'

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

# --- konfiguration ---
d_total = X.shape[1]
# Sæt d_team hvis X er [home | away | (ev. clash)]
USE_TWOTOWER = True           # sæt False hvis du ikke har separat home/away
d_team = d_total // 2 if USE_TWOTOWER else None

weight_decay = 5e-5
lr = 8e-4
alpha_drop = 0.05

# --- direct normalization ---
def direct_normalize(x, eps=1e-6):
    x = tf.nn.relu(x) + eps
    return x / tf.reduce_sum(x, axis=-1, keepdims=True)

# --- Brier-score (kalibrering) ---
def brier_score(y_true, y_pred):
    return tf.reduce_mean(tf.reduce_sum(tf.square(y_pred - y_true), axis=-1))

# --- inputs ---
inp = keras.Input(shape=(d_total,), name="X")

if USE_TWOTOWER:
    home = layers.Lambda(lambda t: t[:, :d_team], name="home")(inp)
    away = layers.Lambda(lambda t: t[:, d_team:2*d_team], name="away")(inp)
    # delt, meget lille tower
    shared = layers.Dense(
        24, activation="selu",
        kernel_regularizer=regularizers.l2(weight_decay),
        name="phi"
    )
    h = shared(home)
    a = shared(away)
    diff = layers.Subtract()([h, a])
    deep_in = layers.Concatenate()([h, a, diff])               # 24*3 = 72 neuroner ind
else:
    deep_in = inp

# --- Deep-lite (2 små lag, SELU + AlphaDropout) ---
z = layers.Dense(48, activation="selu",
                 kernel_regularizer=regularizers.l2(weight_decay))(deep_in)
z = layers.AlphaDropout(alpha_drop)(z)
z = layers.Dense(24, activation="selu",
                 kernel_regularizer=regularizers.l2(weight_decay))(z)
z = layers.AlphaDropout(alpha_drop)(z)

# --- Wide-del (lineær) ---
wide_logits = layers.Dense(3, use_bias=False, name="wide")(inp)

# --- Head: kombiner wide + deep ---
logits = layers.Dense(3, name="deep_head")(z)
logits = layers.Add(name="logits_sum")([logits, wide_logits])

# --- Direct normalization ---
probs = layers.Lambda(direct_normalize, name="probs")(logits)

model = keras.Model(inp, probs, name="wd_lite_directnorm")

# Optimizer: AdamW hvis muligt
try:
    opt = keras.optimizers.experimental.AdamW(learning_rate=lr, weight_decay=weight_decay)
except:
    opt = keras.optimizers.Adam(learning_rate=lr)

model.compile(
    optimizer=opt,
    loss="categorical_crossentropy",
    metrics=["accuracy", brier_score]
)

early   = callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
plateau = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5)

# Brug validation_split for enkelhed (eller lav et eksplicit split)
history = model.fit(
    X_train, Y_train,
    epochs=150,
    batch_size=512,           # større batch -> hurtigere på denne lille model
    validation_split=0.2,
    callbacks=[early, plateau],
    verbose=1
)

Y_pred_proba = model.predict(X_test, batch_size=1024)

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