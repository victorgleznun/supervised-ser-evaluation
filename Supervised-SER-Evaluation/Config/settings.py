import os

# Ruta base del proyecto (contendrá las subcarpetas Datasets/, Results/, etc.)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Directorios de audio
RAVDESS_DIR = os.path.join(BASE_DIR, "Datasets", "AudiosRavdess")
SAVEE_DIR   = os.path.join(BASE_DIR, "Datasets", "AudiosSAVEE")
TESS_DIR    = os.path.join(BASE_DIR, "Datasets", "AudiosTESS")
NEW_AUDIOS  = os.path.join(BASE_DIR, "TestAudios")

# Carpeta donde vuelcan los resultados
RESULTS_DIR = os.path.join(BASE_DIR, "Results")

# Semilla para reproducibilidad
RANDOM_STATE = 42