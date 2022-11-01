import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'

path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)
home_directory = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')

training_directory = os.path.join(home_directory, 'train') # 1000 each
validation_directory = os.path.join(home_directory, 'validation') # 500 each

BATCH_SIZE = 32
IMG_SIZE = (160, 160)

# 2000 dogs and cats，2000 / 32 = 62.5 = 63 batches
train_dataset = tf.keras.utils.image_dataset_from_directory(training_directory, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)

# 1000 dogs and cats, 1000 / 32 = 31.25 = 32 batches
validation_dataset = tf.keras.utils.image_dataset_from_directory(validation_directory, shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE)

# 0 for 'cats', 1 for 'dogs'
class_names = train_dataset.class_names

plt.figure(figsize=(10, 10))

# train_dataset.take(1) takes an entire batch of 32 images and their matching labels
# the batch of 32 images is an EagerTensor of 32 x 160 x 160 x 3
# the matching labels is an EagerTensor of 32
for images, labels in train_dataset.take(1):
  for i in range(32):
    ax = plt.subplot(4, 8, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
plt.show()

val_batches = tf.data.experimental.cardinality(validation_dataset) # 32 batches

# first 6 are for testing, and the rest are for validation
test_dataset = validation_dataset.take(val_batches // 5)
validation_dataset = validation_dataset.skip(val_batches // 5)

print('Number of validation batches: %d' % tf.data.experimental.cardinality(validation_dataset))
print('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset))

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

data_augmentation = tf.keras.Sequential([tf.keras.layers.RandomFlip('horizontal'), tf.keras.layers.RandomRotation(0.2)])

for image, _ in train_dataset.take(1):  # get 1 batch, 32 images with 32 labels
  plt.figure(figsize=(10, 10))
  first_image = image[0]
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    augmented_image = data_augmentation (tf.expand_dims(first_image, 0))
    plt.imshow(augmented_image[0] / 255)
    plt.axis('off')

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
print(preprocess_input) # the function from the model, we will not use it

rescale = tf.keras.layers.Rescaling(1./127.5, offset=-1)

# Create the base model from the pre-trained model MobileNet V2
# (160, 160, 3)
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')


image_batch, label_batch = next( iter(train_dataset) ) # Get a batch - EagerTesnor 32x 160x160  x 3
feature_batch = base_model(image_batch) # 32 x 160x160x3 -> 32 x 5x5x1280


