import os
from PIL import Image
import numpy as np

import tensorflow as tf
from tensorflow.keras.applications.resnet import preprocess_input

IMG_SIZE = (224, 224)

def load_images_for_resnet(data_dir):
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    X, y = [], []

    for label_index, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        for filename in os.listdir(class_dir):
            if filename.lower().endswith((".png", ".jpg", ".jpeg", ".heic")):
                file_path = os.path.join(class_dir, filename)
                try:
                    if filename.lower().endswith(".heic"):
                        print(f"heic is not supported yet: {filename}")
                        continue
                    
                    img = Image.open(file_path).convert("RGB")
                    img = img.resize(IMG_SIZE)
                    img_array = np.array(img)
                    img_array = preprocess_input(img_array)
                    X.append(img_array)
                    y.append(label_index)
                except Exception as e:
                    print(f"Failed to load {filename}: {e}")

    X = np.array(X, dtype=np.float32)
    y = tf.keras.utils.to_categorical(y, num_classes=len(class_names))
    return X, y, class_names

def load_unlabeled_images(data_dir):
    X = []

    for filename in os.listdir(data_dir):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".heic")):
            file_path = os.path.join(data_dir, filename)
            try:
                if filename.lower().endswith(".heic"):
                    print(f"heic is not supported yet: {filename}")
                    continue
                
                img = Image.open(file_path).convert("RGB")
                img = img.resize(IMG_SIZE)
                img_array = np.array(img)
                img_array = preprocess_input(img_array)
                X.append(img_array)
            except Exception as e:
                print(f"Failed to load {filename}: {e}")

    X = np.array(X, dtype=np.float32)
    return X