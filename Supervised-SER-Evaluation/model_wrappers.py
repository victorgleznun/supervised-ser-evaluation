import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from Config.settings import RANDOM_STATE

class BaseWrapper:
    """Base wrapper que gestiona LabelEncoder y define interfaz."""
    def __init__(self):
        self.le = LabelEncoder()
        self.name = None
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Ajusta el LabelEncoder y entrena el modelo."""
        y_enc = self.le.fit_transform(y)
        self.model.fit(X, y_enc)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predice y devuelve etiquetas originales."""
        preds_enc = self.model.predict(X)
        return self.le.inverse_transform(preds_enc)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predice probabilidades (solo si el modelo lo soporta)."""
        return self.model.predict_proba(X)


class HistGBWrapper(BaseWrapper):
    """Wrapper para HistGradientBoostingClassifier."""
    def __init__(self,
                 max_iter=171,
                 max_depth=10,
                 min_samples_leaf=20,
                 random_state=None):
        super().__init__()
        self.name = 'histgb'
        self.model = HistGradientBoostingClassifier(
            max_iter=max_iter,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state or RANDOM_STATE,
        )


class RandomForestWrapper(BaseWrapper):
    """Wrapper para RandomForestClassifier."""
    def __init__(self,
                 n_estimators=100,
                 max_depth=20,
                 min_samples_split=2,
                 class_weight='balanced',
                 random_state=None):
        super().__init__()
        self.name = 'rf'
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            class_weight=class_weight,
            random_state=random_state or RANDOM_STATE
        )


class XGBoostWrapper(BaseWrapper):
    """Wrapper para XGBClassifier, desactiva use_label_encoder."""
    def __init__(self,
                 colsample_bytree=0.8,
                 learning_rate=0.1,
                 max_depth=6,
                 n_estimators=200,
                 subsample=0.8,
                 random_state=None):
        super().__init__()
        self.name = 'xgb'
        self.model = XGBClassifier(
            colsample_bytree=colsample_bytree,
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=n_estimators,
            subsample=subsample,
            random_state=random_state or RANDOM_STATE
        )


class LinearSVCWrapper(BaseWrapper):
    """Wrapper para LinearSVC, incluye parámetro C."""
    def __init__(self,
                 C=0.01,
                 class_weight='balanced',
                 random_state=None):
        super().__init__()
        self.name = 'svc'
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('svc', LinearSVC(
            C=C,
            class_weight=class_weight,
            random_state=random_state or RANDOM_STATE
            ))
        ])


class MLPWrapper(BaseWrapper):
    """Wrapper para MLPClassifier con escalado automático."""
    def __init__(self,
                 hidden_layer_sizes=(128,64),
                 max_iter=1000,
                 alpha=0.0001,
                 learning_rate_init=0.001,
                 solver='adam',
                 tol=1e-6,
                 random_state=None):
        super().__init__()
        self.name = 'mlp'
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                max_iter=max_iter,
                alpha=alpha,
                learning_rate_init=learning_rate_init,
                solver=solver,
                tol=tol,
                random_state=random_state or RANDOM_STATE
            ))
        ])
