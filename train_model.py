# train_model.py
import os
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# === CONFIG ===
BASE_DIR = r"C:\users\harsh\Downloads\TrainingData\dataset_split"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 100
BEST_MODEL_PATH = "best_model_val_acc.h5"

# === LOAD DATA ===
datagen = ImageDataGenerator(rescale=1./255)

train_loader = datagen.flow_from_directory(
    os.path.join(BASE_DIR, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_loader = datagen.flow_from_directory(
    os.path.join(BASE_DIR, "val"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# === BUILD MODEL ===
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
x = GlobalAveragePooling2D()(base_model.output)
output = Dense(train_loader.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

# === COMPILE MODEL ===
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# === CALLBACKS ===
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    BEST_MODEL_PATH,
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=False,
    mode='max',
    verbose=1
)

# === TRAIN MODEL ===
history = model.fit(
    train_loader,
    validation_data=val_loader,
    epochs=EPOCHS,
    callbacks=[early_stop, checkpoint]
)

print(f"✅ Best model saved to: {BEST_MODEL_PATH}")

# === PLOT ACCURACY ===
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.grid(True)
plt.show()
