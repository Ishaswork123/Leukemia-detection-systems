import os
import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = r"E:\leuko-fixed (1)\leuko-main\model\fine_tuned_densenet_leukemia (1).h5"
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

def find_last_conv_layer(model):
    if hasattr(model, "layers") and hasattr(model.layers[0], "layers"):
        search_layers = model.layers[0].layers
    else:
        search_layers = model.layers

    for layer in reversed(search_layers):
        if len(layer.output.shape) == 4:
            return layer.name
    raise ValueError("No convolutional layer found!")

try:
    last_conv_layer_name = find_last_conv_layer(model)
    print("Found last conv layer:", last_conv_layer_name)
    
    # Try to make gradcam heatmap model
    print("Creating grad_model...")
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    print("grad_model created successfully.")
except Exception as e:
    import traceback
    traceback.print_exc()
