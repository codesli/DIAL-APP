import os
import tensorflow as tf
from tensorflow.keras import layers, models

# PATH TO YOUR DATASET — set this via an environment variable so the script
# runs on any machine, not just the one it was written on:
#   export DATASET_DIR="/path/to/C-NMC_Leukemia/training_data/fold_2"
data_dir = os.environ.get(
    'DATASET_DIR',
    os.path.join(os.path.dirname(__file__), 'dataset', 'fold_2')
)

if not os.path.isdir(data_dir):
    raise FileNotFoundError(
        f"Dataset folder not found at: {data_dir}\n"
        "Set the DATASET_DIR environment variable to point at your "
        "training_data/fold_2 folder, e.g.:\n"
        "  export DATASET_DIR=\"/path/to/C-NMC_Leukemia/training_data/fold_2\""
    )

# Load the images (80% for training, 20% for testing during the run)
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(180, 180),
    batch_size=32
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(180, 180),
    batch_size=32
)

# This tells the AI that 'all' is Leukemia and 'hem' is Healthy
class_names = train_ds.class_names

# Build the AI model architecture
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(180, 180, 3)),
    layers.Conv2D(16, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(class_names))
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

print(f"Training on classes: {class_names}")
model.fit(train_ds, validation_data=val_ds, epochs=10)

# Save the model into ./model so app.py finds it automatically
model_save_path = os.environ.get(
    'MODEL_PATH',
    os.path.join(os.path.dirname(__file__), 'model', 'leukemia_model.h5')
)
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
model.save(model_save_path)
print(f"Training finished! Model saved at: {model_save_path}")
