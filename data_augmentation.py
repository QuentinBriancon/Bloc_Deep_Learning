import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, save_img
import numpy as np
import tensorflow as tf

def augment_image(image_path, output_dir, augmentations, save_prefix, num_augmented=1):
    # Load the image
    img = load_img(image_path)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Apply augmentations
    aug_iter = augmentations.flow(img_array, batch_size=1, save_to_dir=output_dir, 
                                  save_prefix=save_prefix, save_format='jpg')
    
    # Generate and save the augmented images
    for i in range(num_augmented):
        next(aug_iter)

def adjust_contrast(img):
    # Apply contrast adjustment (random between 0.8 and 1.2)
    contrast_factor = tf.random.uniform([], 0.8, 1.2)
    img = tf.image.adjust_contrast(img, contrast_factor)
    return img

def augment_dataset(dataset_dir, output_dir, augmentations, num_augmented=1, augment_every_n=1):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over the classes in the dataset
    for class_name in os.listdir(dataset_dir):
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        output_class_dir = os.path.join(output_dir, class_name)
        if not os.path.exists(output_class_dir):
            os.makedirs(output_class_dir)

        # Iterate over the images in the class directory
        for i, image_name in enumerate(os.listdir(class_dir)):
            image_path = os.path.join(class_dir, image_name)

            # Perform augmentation only for every nth image
            if i % augment_every_n == 0:
                save_prefix = os.path.splitext(image_name)[0]
                augment_image(image_path, output_class_dir, augmentations, save_prefix, num_augmented)

if __name__ == "__main__":
    # Paths to your dataset and output directory
    dataset_dir = '/mnt/c//Users/quent/Documents/Devoir/CESI/Deep learning/OneDrive_2025-03-26/Data Science/Datasets/Dataset Livrable 1 - PreCleanForAugmentation'
    output_dir = '/mnt/c//Users/quent/Documents/Devoir/CESI/Deep learning/OneDrive_2025-03-26/Data Science/Datasets/Dataset Livrable 1 - Augmented'

    # Define your augmentation parameters
    augmentations = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        preprocessing_function=adjust_contrast,
        fill_mode='constant',  # Ne remplit pas avec les bords
        cval=255,  
    )

    # Augment every image and create 5 augmented versions
    augment_dataset(dataset_dir, output_dir, augmentations, num_augmented=4, augment_every_n=1)
