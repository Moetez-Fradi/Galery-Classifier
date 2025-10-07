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

def unlabeled_image_generator(data_dir, batch_size=32):
    filenames = [f for f in os.listdir(data_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    while True:
        np.random.shuffle(filenames)
        for i in range(0, len(filenames), batch_size):
            batch_files = filenames[i:i+batch_size]
            batch_imgs = []
            for f in batch_files:
                img_path = os.path.join(data_dir, f)
                img = Image.open(img_path).convert("RGB").resize((224,224))
                arr = preprocess_input(np.array(img))
                batch_imgs.append(arr)
            yield np.array(batch_imgs, dtype=np.float32), np.array(batch_files)
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