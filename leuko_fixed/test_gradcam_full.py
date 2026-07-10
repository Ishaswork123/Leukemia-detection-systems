import os
import cv2
import numpy as np
import tensorflow as tf
from api.index import generate_and_save_gradcam, get_model, predict_with_tta

MODEL_PATH = r"E:\leuko-fixed (1)\leuko-main\model\fine_tuned_densenet_leukemia (1).h5"
print("Loading model...")
model = get_model()
print("Model loaded.")

# create a dummy image
test_img_path = "test_dummy.jpg"
dummy_img = np.zeros((400, 400, 3), dtype=np.uint8)
cv2.imwrite(test_img_path, dummy_img)

save_path = "test_heatmap.jpg"

try:
    label, conf, class_idx = predict_with_tta(test_img_path)
    print(f"Predicted class_idx: {class_idx}")
    success = generate_and_save_gradcam(test_img_path, model, class_idx, save_path)
    print("Success:", success)
except Exception as e:
    import traceback
    traceback.print_exc()

if os.path.exists(test_img_path): os.remove(test_img_path)
