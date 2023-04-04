import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('face_mask_classification.h5')

# Load the image from user input
img_path = input('Enter path to image file: ')
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

# Preprocess the image
img_array /= 255.

# Use the model to make a prediction
prediction = model.predict(img_array)

# Print the predicted class and probability
class_names = ['cloth', 'n95', 'n95v', 'nfm', 'srg'] # replace with your class names
predicted_class = class_names[np.argmax(prediction)]
predicted_prob = np.max(prediction)

print(f'Predicted class: {predicted_class}')
print(f'Probability: {predicted_prob}')
