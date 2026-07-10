import os
import cv2
import numpy as np
import tensorflow as tf
from api.index import get_model, predict_with_tta

MODEL_PATH = r"E:\leuko-fixed (1)\leuko-main\model\fine_tuned_densenet_leukemia (1).h5"
model = get_model()

def test_heatmap():
    test_img = np.zeros((1, 224, 224, 3), dtype=np.float32)
    last_conv_layer_name = "relu" # we found this earlier
    print("creating grad model")
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    print("test inputs")
    with tf.GradientTape() as tape:
        res = grad_model(test_img)
        print("res length:", len(res), "types:", [type(r) for r in res])
        conv_outputs, predictions = res
        print("predictions type:", type(predictions), "shape:", getattr(predictions, 'shape', 'NO_SHAPE'))
        pred_index = 4
        try:
            class_channel = predictions[:, pred_index]
            print("class_channel shape:", getattr(class_channel, 'shape', 'NO_SHAPE'))
        except Exception as e:
            print("Error on predictions[:, pred_index]:", e)

        try:
            grads = tape.gradient(class_channel, conv_outputs)
            print("grads type:", type(grads))
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            print("pooled type:", type(pooled_grads))
            heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
            print("heatmap type:", type(heatmap))
        except Exception as e:
            print("Error on grads/heatmap computing:", e)

test_heatmap()
