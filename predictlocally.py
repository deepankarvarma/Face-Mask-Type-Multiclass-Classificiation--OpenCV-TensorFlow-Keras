import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model('face_mask_classification.h5')



# Load the image and preprocess it
img = image.load_img("images/5.jpg", target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.

# Make a prediction using the model
prediction = model.predict(img_array)

# Get the predicted class and probability
class_names = ['cloth', 'n95', 'n95v', 'nfm', 'srg'] # replace with your class names
predicted_class = class_names[np.argmax(prediction)]
predicted_prob = np.max(prediction)

# Display the input image and predicted class
plt.imshow(img)
plt.title(f'Predicted class: {predicted_class}, Probability: {predicted_prob}')
plt.axis('off')
plt.show() 