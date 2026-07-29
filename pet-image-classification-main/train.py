import tensorflow as tf

# Load Dataset
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset",
    image_size=(128,128),
    batch_size=8,
    validation_split=0.2,
    subset="training",
    seed=42
)

val_data = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset",
    image_size=(128,128),
    batch_size=8,
    validation_split=0.2,
    subset="validation",
    seed=42
)

# CNN Model
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),

    tf.keras.layers.Conv2D(32,(3,3),activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64,(3,3),activation="relu"),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128,activation="relu"),

    tf.keras.layers.Dense(1,activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(train_data, validation_data=val_data, epochs=100)

model.save("cat_dog_model.keras")

print("Model Saved Successfully")

