import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import random

#  Charger une image et la convertir en niveaux de gris ou RGB
def load_image(path, grayscale=False):
    image = Image.open(path)
    if grayscale:
        image = image.convert('L')  # Convertir en niveaux de gris
    else:
        image = image.convert('RGB')  # Convertir en RGB
    return np.array(image, dtype=np.float32) / 255.0  # Normaliser [0, 1]


# 2. Appliquer une convolution avec un noyau donné

def apply_filter(image, kernel):
    kernel = np.array(kernel, dtype=np.float32)
    
    if image.ndim == 2:
        # Image en niveaux de gris
        image = tf.expand_dims(image, axis=-1)  # [H, W] → [H, W, 1]
        kernel_tf = tf.reshape(kernel, [kernel.shape[0], kernel.shape[1], 1, 1])
    else:
        # Image RGB
        channels = 3
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        kernel_tf = tf.stack([kernel] * channels, axis=-1)  # [H, W, C]
        kernel_tf = tf.reshape(kernel_tf, [kernel.shape[0], kernel.shape[1], channels, 1])

    # Ajouter la dimension batch
    image = tf.expand_dims(image, axis=0)  # [1, H, W, C]

    # Appliquer la convolution
    result = tf.nn.conv2d(image, filters=kernel_tf, strides=1, padding='SAME')

    # Retirer dimensions batch
    result = tf.squeeze(result)

    return result.numpy()

# 3. Fonctions pour générer des filtres prédéfinis

def blur_kernel():
    return np.ones((3, 3)) / 9.0

def sobel_horizontal():
    return np.array([[1, 2, 1],
                     [0, 0, 0],
                     [-1, -2, -1]], dtype=np.float32)

def sobel_vertical():
    return np.array([[1, 0, -1],
                     [2, 0, -2],
                     [1, 0, -1]], dtype=np.float32)

# 4. Générer un filtre aléatoire

def random_kernel(size):
    kernel = np.random.uniform(-1, 1, (size, size))
    kernel /= np.sum(np.abs(kernel))  # Normalisation pour éviter trop de saturation
    return kernel.astype(np.float32)

# 5. Afficher les images

def show_images(images, titles):
    plt.figure(figsize=(15, 5))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, len(images), i+1)
        cmap = 'gray' if img.ndim == 2 else None
        plt.imshow(img, cmap=cmap)
        plt.title(title)
        plt.axis('off')
    plt.show()

# 6. Exécution principale

if __name__ == "__main__":
    path = 'lena.png'  # Remplace par le chemin de ton image
    is_gray = False  # True pour niveaux de gris

    img = load_image(path, grayscale=is_gray)

    filters = {
        "Original": None,
        "Flou": blur_kernel(),
        "Sobel H": sobel_horizontal(),
        "Sobel V": sobel_vertical(),
        "Random 3x3": random_kernel(3),
        "Random 5x5": random_kernel(5),
        "Random 7x7": random_kernel(7),
    }

    images = [img]
    titles = ["Original"]

    for name, kernel in filters.items():
        if kernel is not None:
            result = apply_filter(img, kernel)
            images.append(result)
            titles.append(name)

    show_images(images, titles)
