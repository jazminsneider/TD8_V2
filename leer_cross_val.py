#!/usr/bin/env python
# coding: utf-8
import os
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
#RESULTS_FOLDER = "Results/F1_OVER"
RESULTS_FOLDER = "Results/F1_NO_OVER"  # carpeta donde guardaste los fold
Y_TRUE_FILE = os.path.join(RESULTS_FOLDER, "y.csv")

# Leemos las etiquetas verdaderas
y_true_df = pd.read_csv(Y_TRUE_FILE, index_col=0)
y_true = y_true_df.values.squeeze()

# Inicializamos variables
all_preds = pd.DataFrame(index=y_true_df.index)

# Recorremos todas las carpetas de fold
for fold_name in os.listdir(RESULTS_FOLDER):
    fold_path = os.path.join(RESULTS_FOLDER, fold_name)
    val_probas_file = os.path.join(fold_path, "val_probas.csv")
    if os.path.exists(val_probas_file):
        fold_probs = pd.read_csv(val_probas_file, index_col=0)
        # Predicción de clase como la que tiene mayor probabilidad
        fold_preds = fold_probs.idxmax(axis=1)
        all_preds = all_preds.join(fold_preds.rename(fold_name), how='left')

# Tomamos la primera predicción no nula de cada fila (debería haber solo una por fold)
final_preds = all_preds.bfill(axis=1).iloc[:, 0].values

# Calculamos F1
labels = np.unique(y_true)
f1_per_class = f1_score(y_true, final_preds, average=None, labels=labels)
f1_macro = f1_score(y_true, final_preds, average='macro')

print("F1 por clase:")
for lbl, f1_val in zip(labels, f1_per_class):
    print(f"Clase {lbl}: {f1_val:.4f}")

print(f"\nF1 macro total: {f1_macro:.4f}")
