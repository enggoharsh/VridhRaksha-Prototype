# data_loader.py
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(base_dir, img_size=(224, 224), batch_size=32):
    datagen = ImageDataGenerator(rescale=1./255)

    train_loader = datagen.flow_from_directory(
        os.path.join(base_dir, "train"),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    val_loader = datagen.flow_from_directory(
        os.path.join(base_dir, "val"),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_loader = datagen.flow_from_directory(
        os.path.join(base_dir, "test"),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    return train_loader, val_loader, test_loader
