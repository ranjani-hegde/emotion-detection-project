import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("emotion_model.h5", compile=False)

emotions = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera not working")
    exit()

cv2.namedWindow("Emotion Detector", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()

    if not ret:
        print("❌ Frame not captured")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face,(48,48))
        face = face/255.0
        face = face.reshape(1,48,48,1)

        pred = model.predict(face, verbose=0)
        label = emotions[np.argmax(pred)]

        cv2.putText(frame,label,(x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

    cv2.imshow("Emotion Detector", frame)

    # VERY IMPORTANT
    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()