import os
from PIL import Image
from IPython.display import display

def load_images_from_directory(directory_path):
    supported_extensions = (".png", ".jpg", ".jpeg", ".heic")
    images = []

    for filename in os.listdir(directory_path):
        if filename.lower().endswith(supported_extensions):
            file_path = os.path.join(directory_path, filename)
            try:
                if filename.lower().endswith(".heic"):
                    pass
                else:
                    img = Image.open(file_path)
                images.append(img)
            except Exception as e:
                print(f"Failed to load {filename}: {e}")
    return images