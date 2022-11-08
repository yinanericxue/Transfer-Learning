# Transfer-Learning

# There is a major problem with the Dog and Cat classification project, which is the insufficient amount of training data despite after performing image augmentation. To fix this, I must use transfer learning, which is the process of applying a pre-trained model and not start from scratch.

# There are two ways to customize a pre-trained model: 1) Feature Extraction, the process of loading an entire pre-trained convolutional base (the entire neural network) and only changing the bottleneck and classifying layers towards the end. 2) Fine-Tuning, the process to unfreeze layers inside the convolutional base and allow them to be trained and updated. This typically should be done to layers toward the end since they represent higher-order features and are more relevant for the specific task.

# To use Google’s MobileNet V2, a base model pre-trained by ImageNet, I first need to rescale all pixel values of my images to this model's requirement from [0, 255] to [-1,1]. 
# rescale = tf.keras.layers.Rescaling(1./127.5, offset=-1)

# To begin the Feature Extraction customization, I first instantiate the model and remove its former classification layers.
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
