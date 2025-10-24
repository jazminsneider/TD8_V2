import helper
import pandas as pd
import os
import numpy as np
from sklearn import preprocessing
import numpy as np
import os.path
import click
import pandas as pd
from tqdm import tqdm
import helper
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE
from collections import Counter
import sklearn.metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

np.random.seed(1234)
def get_class_weights(y):
    counter = Counter(y)
    majority = max(counter.values())
    return {cls: round(float(majority) / float(count), 2) for cls, count in counter.items()}

def apply_resampling(balance_method, X_train, y_train):
    weights = get_class_weights(y_train)
    weights = get_class_weights(y_train)
    if balance_method == "oversample":
        resampling_indices, _ = RandomOverSampler().fit_resample(X=np.arange(len(y_train)).reshape(-1, 1), y=y_train)
        X_train, y_train = X_train[resampling_indices.squeeze()], y_train[resampling_indices.squeeze()]

    if balance_method == "smote":
        if min([v for (c, v) in Counter(y_train).items()]) < 6:
            for classs, count in Counter(y_train).items():
                if count < 6:
                    helper.warning("could not apply SMOTE directly (#instances={}), RUNNING OVERSAMPLER FIRST".format(Counter(y_train)))
                    resampling_indices = np.concatenate([np.arange(len(y_train)), np.arange(len(y_train))[y_train == classs].repeat(6 // count)])
                    X_train, y_train = X_train[resampling_indices], y_train[resampling_indices]

        resampling_indices, _ = SMOTE().fit_resample(X=np.arange(len(y_train)).reshape(-1, 1), y=y_train)
        X_train, y_train = X_train[resampling_indices.squeeze()], y_train[resampling_indices.squeeze()]

    if balance_method != "class_weights":
        weights = None
    return weights, X_train, y_train



FILL_NA_WITH = -15
N_ESTIMATORS = 300
MAX_DEPTH = 10
MAX_FEATURES = 0.5

output_fname = "cross_val_resultados_overlap_real.csv"
data_folder = "X_Y/overlap/dev/"
balance_method = "oversample"


X = pd.read_csv(os.path.join(data_folder, "X.csv"), index_col=0)

X = X.fillna(FILL_NA_WITH).values.astype(np.float32)
y_true = pd.read_csv(os.path.join(data_folder, "y.csv"), index_col=0)
sessions= pd.read_csv(os.path.join(data_folder, "sessions.csv"), index_col=0).values.squeeze()

 
labels_keep = ["BC_O", "BI", "I", "O", "X2_O"]
#labels_keep = ["PI", "BC", "S", "X2"] #Para no overlap
mask = y_true.isin(labels_keep).values.flatten()

X = X[mask]               # filtra X
y_true = y_true[mask]            # filtra y_true
sessions = sessions[mask] # filtra sesiones
print(f"Quedaron {len(y_true)} ejemplos con las etiquetas {y_true.value_counts()}")

le = preprocessing.LabelEncoder()
y = le.fit_transform(y_true.values.squeeze())

group_cross_val = LeaveOneGroupOut()
results = []


combined_y_pred=[]
combined_y_true=[]
train_sizes=[]
res_i={}
for (train_positions, val_positions) in tqdm(group_cross_val.split(X, y, groups=sessions), total=group_cross_val.get_n_splits(groups=sessions)):
            val_sess = set(sessions[val_positions])
            assert len(val_sess) == 1

            X_train = X[train_positions]
            y_train = y[train_positions]

            X_val = X[val_positions]
            y_val = y[val_positions]

            class_weights, X_train_res, y_train_res = apply_resampling(balance_method, X_train, y_train)

            # helper.info(val_sess, X_train.shape, y_train.shape)
            # helper.info(val_sess, X_train_res.shape, y_train_res.shape)

            clf = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, class_weight=class_weights, max_features=MAX_FEATURES, n_jobs=5)

            clf.fit(X_train_res, y_train_res)
            y_probas_val = clf.predict_proba(X_val)
            y_pred = pd.DataFrame(y_probas_val, columns=[le.classes_[c] for c in clf.classes_])  # index=idx_val,
            for c in le.classes_:
                if c not in y_pred.columns:
                    y_pred[c] = 0.

            y_pred = y_pred.idxmax(axis=1)
            y_pred = le.transform(y_pred)

            combined_y_pred.extend(y_pred)
            combined_y_true.extend(y_val)
            train_sizes.append(len(y_train))


f1 = sklearn.metrics.f1_score(y_true=combined_y_true, y_pred=combined_y_pred, average="macro")
helper.info(f1)
res_i["f1_macro (global)"] = f1
res_i["train_size_mean"] = np.mean(train_sizes)
results.append(res_i)

results_df = pd.DataFrame(results)
results_df.to_csv(output_fname, index=False)



# matriz de confusión
cm = confusion_matrix(combined_y_true, combined_y_pred, labels=np.arange(len(le.classes_)))

# Graficar
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_,
            yticklabels=le.classes_)

plt.xlabel("Predicción")
plt.ylabel("Etiqueta real")
plt.title("Matriz de confusión - Random Forest")
plt.tight_layout()
plt.show()

# 🔹 Calcular F1-score por clase usando todas las predicciones acumuladas
f1_per_class = sklearn.metrics.f1_score(
    y_true=combined_y_true,
    y_pred=combined_y_pred,
    average=None,
    labels=np.arange(len(le.classes_))
)

print("\nF1-score por clase:")
for label, f1_val in zip(le.classes_, f1_per_class):
    print(f"  {label}: {f1_val:.4f}")
