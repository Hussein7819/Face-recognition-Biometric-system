# evaluation.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve


def plot_gen_imp(genuine_scores, imposter_scores, method_name):
    """Plots the Genuine and Imposter Distribution (Gen-Imp Curve)."""
    plt.figure(figsize=(10, 5))
    plt.hist(genuine_scores, bins=30, alpha=0.5, label='Genuine', color='blue', density=True)
    plt.hist(imposter_scores, bins=30, alpha=0.5, label='Imposter', color='red', density=True)
    plt.title(f'Gen-Imp Distribution: {method_name}')
    plt.xlabel('Distance')
    plt.ylabel('Density')
    plt.legend()
    plt.show()


def calculate_metrics(genuine_scores, imposter_scores, method_name):
    """Calculates D-prime, EER, and TMR at specific FMRs."""

    # 1. D-prime calculation
    mu_gen = np.mean(genuine_scores)
    mu_imp = np.mean(imposter_scores)
    std_gen = np.std(genuine_scores)
    std_imp = np.std(imposter_scores)

    d_prime = abs(mu_gen - mu_imp) / np.sqrt(0.5 * (std_gen ** 2 + std_imp ** 2)) 

    # 2. ROC and EER
    # We use -scores because roc_curve expects 'similarity' or 'probability'
    # (higher is better), but we have 'distances' (lower is better).
    labels = [1] * len(genuine_scores) + [0] * len(imposter_scores)
    scores = list(-genuine_scores) + list(-imposter_scores)
    fpr, tpr, thresholds = roc_curve(labels, scores)

    # EER calculation
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    # 3. TMR (1-FRR) at specific FMR (FPR)
    # TMR at 1% FMR (0.01) and 0.1% FMR (0.001)
    tmr_at_1 = interp1d(fpr, tpr)(0.01)
    tmr_at_01 = interp1d(fpr, tpr)(0.001)



    # 4. Plot ROC
    plt.figure()
    plt.plot(fpr, tpr, label=f'{method_name} (EER = {eer:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Match Rate (FMR)')
    plt.ylabel('True Match Rate (TMR)')
    plt.title(f'ROC Curve: {method_name}')
    plt.legend()
    plt.show()

    return d_prime, eer, tmr_at_1, tmr_at_01