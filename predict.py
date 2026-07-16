import os
import sys
import tensorflow as tf
import numpy as np

MODEL_PATH = os.environ.get(
    'MODEL_PATH',
    os.path.join(os.path.dirname(__file__), 'model', 'leukemia_model.h5')
)
MAP = {0: "Leukemia (ALL)", 1: "Healthy (HEM)"}


def run_prediction(image_path):
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        print(f" Error loading model at {MODEL_PATH}: {e}")
        return

    try:
        img = tf.keras.utils.load_img(image_path, target_size=(180, 180))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        result = MAP[int(np.argmax(score))]
        print(f"\n Result: {result}")
        print(f" Confidence: {100 * np.max(score):.2f}%")
    except Exception as e:
        print(f" Error reading image: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 predict.py [path_to_image]")
    else:
        run_prediction(sys.argv[1])
