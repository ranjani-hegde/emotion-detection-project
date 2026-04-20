import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("emotion_model.h5")

emotions = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

img = cv2.imread("test.jpg", cv2.IMREAD_GRAYSCALE)

img = cv2.resize(img, (48,48))
img = img / 255.0
img = img.reshape(1,48,48,1)

pred = model.predict(img)
print("Predicted Emotion:", emotions[np.argmax(pred)])