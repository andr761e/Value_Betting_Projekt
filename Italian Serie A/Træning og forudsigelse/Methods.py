import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial

class methods:
    def direct_normalization(self, x):
        return x / np.sum(x, axis=1, keepdims=True)

    def polynomial_calibration(self, Y_pred_proba, Y_test, degree=3):
        calibrated_probs = []
        for i in range(Y_pred_proba.shape[1]):
            poly = Polynomial.fit(Y_pred_proba[:, i], Y_test[:, i], deg=degree)
            calibrated_probs.append(poly(Y_pred_proba[:, i]))
        return np.column_stack(calibrated_probs)

    # Plot methods
    def plot_histogram(self, Y_pred_proba, classes, colors):
        plt.figure(figsize=(10, 6))
        for i, cls in enumerate(classes):
            plt.hist(Y_pred_proba[:, i], bins=20, alpha=0.5, label=f"Forudsagt: {cls}", color=colors[i])
        plt.title("Fordeling af forudsagte sandsynligheder pr. klasse")
        plt.xlabel("Sandsynlighed")
        plt.ylabel("Antal kampe")
        plt.legend()
        plt.show()

    def plot_histogram2(self, Y_pred_proba, classes, colors):
        plt.figure(figsize=(10, 6))
        for i, cls in enumerate(classes):
            plt.hist(Y_pred_proba[:, i], bins=20, alpha=0.5, label=f"Forudsagt: {cls}", color=colors[i])
        plt.title("Fordeling af faktiske sandsynligheder pr. klasse")
        plt.xlabel("Sandsynlighed")
        plt.ylabel("Antal kampe")
        plt.legend()
        plt.show()

    def plot_comparison(self, Y_pred_proba, Y_test, classes, colors):
        plt.figure(figsize=(12, 6))
        x = np.arange(10)
        for i, cls in enumerate(classes):
            plt.plot(x, Y_test[:10, i], marker='o', linestyle='--', label=f"Faktisk: {cls}", color=colors[i])
            plt.plot(x, Y_pred_proba[:10, i], marker='x', linestyle='-', label=f"Forudsagt: {cls}", color=colors[i])
        plt.title("Faktiske vs. forudsagte sandsynligheder (første 10 kampe)")
        plt.xlabel("Kamp")
        plt.ylabel("Sandsynlighed")
        plt.xticks(x, [f"Kamp {i+1}" for i in x])
        plt.legend()
        plt.show()

    def plot_correlation(self, Y_pred_proba, Y_test, classes, colors):
        plt.figure(figsize=(10, 6))
        for i, cls in enumerate(classes):
            plt.scatter(Y_test[:, i], Y_pred_proba[:, i], alpha=0.5, label=cls, color=colors[i])
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
        plt.title("Sammenhæng mellem faktiske og forudsagte sandsynligheder")
        plt.xlabel("Faktiske sandsynligheder")
        plt.ylabel("Forudsagte sandsynligheder")
        plt.legend()
        plt.show()

    def plot_log_loss(self, Y_pred_proba, Y_test):
        log_loss_contributions = -np.sum(Y_test * np.log(Y_pred_proba + 1e-15), axis=1)
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(len(Y_test)), log_loss_contributions, marker='o', linestyle='-', color='purple')
        plt.title("Log-loss bidrag pr. kamp")
        plt.xlabel("Kamp ID")
        plt.ylabel("Log-loss bidrag")
        plt.grid()
        plt.show()