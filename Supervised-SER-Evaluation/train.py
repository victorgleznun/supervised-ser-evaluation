import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder

from Config.settings import RANDOM_STATE, RESULTS_DIR, NEW_AUDIOS
from data_utils import load_ravdess, load_savee, load_tess, extract_features
from model_wrappers import (
    HistGBWrapper,
    RandomForestWrapper,
    XGBoostWrapper,
    LinearSVCWrapper,
    MLPWrapper
)
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")


def save_classification_report(report: str,
                               acc: float,
                               importances: np.ndarray,
                               feat_names: list,
                               cv_results: dict,
                               scoring: list,
                               out_dir: str):
    """
    Guarda en metrics.txt:
      - Informe de clasificación
      - Precisión global
      - Resultados de validación cruzada (media ± std) para cada métrica
      - Importancias de característica (Permutation Importance)
    """
    path = os.path.join(out_dir, 'metrics.txt')
    idx = np.argsort(importances)[::-1]

    with open(path, 'w', encoding='utf-8') as f:
        f.write("=== Informe de Clasificación ===\n")
        f.write(report + "\n")
        f.write(f"Precisión Total: {acc:.4f}\n\n")

        f.write("=== Validación cruzada ===\n")
        for m in scoring:
            vals = cv_results[f'test_{m}']
            f.write(f"{m}: {vals.mean():.4f} ± {vals.std():.4f}\n")
        f.write("\n")

        f.write("=== Importancia de Características (Permutation Importance) ===\n")
        for i in idx:
            f.write(f"{feat_names[i]}: {importances[i]:.4f}\n")


def plot_confusion(cm: np.ndarray, labels: list, out_dir: str):
    plt.figure(figsize=(6,5))
    plt.imshow(cm, cmap='Blues')
    plt.title('Matriz de Confusión')
    plt.colorbar()
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            color = 'white' if cm[i,j] > cm.max()/2 else 'black'
            plt.text(j, i, cm[i,j], ha='center', va='center', color=color)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'))
    plt.clf()


def plot_importances(importances: np.ndarray,
                     feat_names: list,
                     out_dir: str):
    idx = np.argsort(importances)[::-1]
    plt.figure(figsize=(8,5))
    plt.bar(range(len(idx)), importances[idx])
    plt.xticks(range(len(idx)), [feat_names[i] for i in idx], rotation=90)
    plt.title('Importancia de Características (Permutation)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'feature_importances.png'))
    plt.clf()


def save_new_predictions(clf, out_dir: str):
    path = os.path.join(out_dir, 'new_predictions.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write("=== Predicción de Audios Nuevos ===\n")
        for fn in os.listdir(NEW_AUDIOS):
            if not fn.lower().endswith('.wav'):
                continue
            feat = extract_features(os.path.join(NEW_AUDIOS, fn))
            pred = clf.predict([feat])[0] if feat is not None else "N/A"
            f.write(f"{fn}: {pred}\n")


def main(args):
    # 1) Cargar los tres datasets
    X1, y1 = load_ravdess()
    X2, y2 = load_savee()
    X3, y3 = load_tess()

    # 2) Concatenar y filtrar etiquetas None
    X = np.vstack([X1, X2, X3])
    y = np.hstack([y1, y2, y3])
    valid = [i for i, lbl in enumerate(y) if lbl is not None]
    X, y = X[valid], y[valid]

    # 3) LabelEncoder único para todo y versión codificada
    le_global = LabelEncoder().fit(y)
    y_enc = le_global.transform(y)

    # 4) División train/test 80–20 estratificada
    X_train, X_test, y_train_enc, y_test_enc = train_test_split(
        X, y_enc, test_size=0.2, stratify=y_enc, random_state=RANDOM_STATE
    )

    # 5) Instanciar el modelo
    if   args.model == 'histgb':
        clf = HistGBWrapper(random_state=RANDOM_STATE)
    elif args.model == 'rf':
        clf = RandomForestWrapper(random_state=RANDOM_STATE)
    elif args.model == 'xgb':
        clf = XGBoostWrapper(random_state=RANDOM_STATE)
    elif args.model == 'svc':
        clf = LinearSVCWrapper(random_state=RANDOM_STATE)
    else:  # 'mlp'
        clf = MLPWrapper(random_state=RANDOM_STATE)

    # 6) Entrenar y predecir
    clf.fit(X_train, le_global.inverse_transform(y_train_enc))
    y_pred = clf.predict(X_test)

    # 7) Informe en consola (convertir y_test_enc a etiquetas originales)
    y_test_labels = le_global.inverse_transform(y_test_enc)
    report = classification_report(y_test_labels, y_pred, target_names=le_global.classes_)
    acc = accuracy_score(y_test_enc, le_global.transform(y_pred))
    print("\n=== Informe de Clasificación ===")
    print(report)
    print(f"Precisión Total: {acc:.4f}")

    # 8) Validación cruzada (5-fold o 10-fold para SVC)
    folds = 10 if args.model == 'svc' else 5
    scoring = ['accuracy','precision_macro','recall_macro','f1_macro']
    cv = cross_validate(clf.model, X, y_enc, cv=folds, scoring=scoring)
    print(f"\nValidación cruzada ({folds}-fold):")
    for m in scoring:
        vals = cv[f'test_{m}']
        print(f"  {m}: {vals.mean():.4f} ± {vals.std():.4f}")

    # 9) Directorio de resultados
    out = os.path.join(RESULTS_DIR, clf.name)
    os.makedirs(out, exist_ok=True)

    # 10) Permutation Importance
    perm = permutation_importance(clf.model, X_test, y_test_enc,
                                  n_repeats=10, random_state=RANDOM_STATE)
    importances = perm.importances_mean
    feat_names = [f"mfcc_{i+1}" for i in range(13)] + \
                 [f"chroma_{i+1}" for i in range(12)] + \
                 ["rmse","zcr","centroid"]

    # 11) Guardar métricas e importances
    save_classification_report(
        report, acc,
        importances, feat_names,
        cv, scoring,
        out
    )

    # 12) Matriz de confusión y gráfico de importances
    cm = confusion_matrix(y_test_labels, y_pred, labels=le_global.classes_)
    plot_confusion(cm, list(le_global.classes_), out)
    plot_importances(importances, feat_names, out)

    # 13) Predicción de audios nuevos
    save_new_predictions(clf, out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Entrena y evalúa modelos de reconocimiento emocional por voz"
    )
    parser.add_argument('--model',
                        choices=['histgb','rf','xgb','svc','mlp'],
                        required=True,
                        help="Modelo a ejecutar")
    args = parser.parse_args()
    main(args)
