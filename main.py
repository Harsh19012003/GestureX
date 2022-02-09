# MINI PROJECT
# HAND GESTURE RECOGNITION

import os
import cv2
from tensorflow.keras.models import load_model
# from keras_visualizer import visualizer

# Loading model
model = load_model("model_0.h5")

model.summary()

# Visualizing model
# visualizer(model, format='png', view=True)


cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    x1 = 100
    y1 = 100
    x2 = 350
    y2 = 350

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

    cv2.imshow("Frame", frame)

    roi = frame[y1:y2, x1:x2]

    roi = cv2.resize(roi, (64, 64))

    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    _, roi = cv2.threshold(roi, 150, 255, cv2.THRESH_BINARY)

    cv2.imshow("ROI", roi)

    img_tobe_predicted = roi.reshape(1, 64, 64, 1)

    result = model.predict(img_tobe_predicted)

    print(result)

    if cv2.waitKey(10) & 0xFF == 27:
        break
   
    
cap.release()
cv2.destroyWindow()
