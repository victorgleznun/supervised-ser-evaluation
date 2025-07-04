# 🎙️ Supervised-SER-Evaluation
Sistema de reconocimiento de emociones en el habla desarrollado como parte de mi Trabajo de Fin de Grado. Compara cinco modelos supervisados (MLP, Random Forest, XGBoost, HistGradientBoosting y LinearSVC) a partir de características acústicas extraídas de voz. Incluye entrenamiento, predicción de audios, informes y gráficos resultantes


## 📚 Descripción general

Supervised-SER-Evaluation permite:

- Entrenar cinco modelos distintos (MLP, Random Forest, XGBoost, HistGradientBoosting y LinearSVC).
- Validar los resultados con técnicas como k-fold cross-validation.
- Realizar predicciones sobre audios externos añadidos por el usuario.
- Exportar automáticamente informes, gráficas y resultados a carpetas organizadas por modelo.

---

## 🧠 Modelos disponibles

Comando para el inicio del sistema: 
```bash
py train.py --model X

Puedes seleccionar cualquiera de los siguientes modelos con el argumento `--model`:

- `mlp` → MultiLayer Perceptron Classifier
- `rf` → Random Forest Classifier
- `xgb` → Extreme Gradient Boost Classifier
- `histgb` → Hist Gradient Boosting Classifier
- `svc` → Linear Support Vector Classifier

---

## 🧾 Requisitos

- Python 3.12

---

### 📦 Instalación de dependencias

```bash
py -m pip install (*)

(*) Dependencias a instalar:
- numpy
- librosa
- tqdm
- scikit-learn
- xgboost
- matplotlib
