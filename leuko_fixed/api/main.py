from pathlib import Path
import tensorflow as tf

# Folder containing this script (api/)
BASE_DIR = Path(__file__).resolve().parent

# Project root (leuko_fixed/)
PROJECT_ROOT = BASE_DIR.parent

# Model path
MODEL_PATH = PROJECT_ROOT / "model" / "fine_tuned_densenet_leukemia (1).h5"

print(MODEL_PATH)
print(MODEL_PATH.exists())

model = tf.keras.models.load_model(MODEL_PATH,compile=False)

model.save(PROJECT_ROOT / "model" / "fine_tuned_densenet_leukemia.keras")

print("Done!")
