import tensorflow as tf
print("--- ENVIRONMENT IS OPEN ---")
print(f"TensorFlow is running version: {tf.__version__}")
if tf.config.list_physical_devices('GPU'):
    print("ESLI ZAIMI")
else:
    print("Aspa me duket se e. rregullova")
    