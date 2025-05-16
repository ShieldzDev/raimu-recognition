from tensorflow.keras.models import load_model
from PIL import Image
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN

import keras
# Load model
class MCDropout(keras.layers.Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True)

model = load_model("./models/my_model.h5", custom_objects={"MCDropout": MCDropout})
# MTCNN detector
detector = MTCNN()

# Image size
IMG_SIZE = 160

# Class label mapping (optional if you want readable output)
class_names = ['surprised', 'sleepy', 'happy', 'leftlight', 'noglasses', 'rightlight', 'glasses', 'wink', 'normal', 'centerlight', 'sad']


def preprocess_face(face, target_size=(IMG_SIZE, IMG_SIZE)):
    image = Image.fromarray(face)
    image = image.convert('L')  # Convert to grayscale
    image = image.resize(target_size)
    img_array = np.array(image).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel
    img_array = np.expand_dims(img_array, axis=0)   # Add batch
    return img_array

cap = cv2.VideoCapture(0)  # or path to video file

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(rgb_frame)

    for result in results:
        x, y, w, h = result['box']
        x, y = abs(x), abs(y)
        face = rgb_frame[y:y+h, x:x+w]

        try:
            face_input = preprocess_face(face)
            prediction = model.predict(face_input)
            label = np.argmax(prediction)
            confidence = np.max(prediction)

            # Draw bounding box and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_names[label]}: {confidence:.2f}", 
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.9, (0, 255, 0), 2)
        except Exception as e:
            print("Face preprocessing error:", e)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
