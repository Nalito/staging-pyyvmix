# Edited get_inference.py for Mini-Xception (grayscale, 48x48)
import tensorflow as tf
import cv2
import numpy as np
import os

class FramePredictor:
    def __init__(self, model_path, frames_folder):
        self.model = tf.keras.models.load_model(model_path)
        self.frames_folder = frames_folder
        self.classnames = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    def preprocess_image(self, img_path):
        # Load in grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (48, 48))
        img = img / 255.0  # Normalize
        img = np.expand_dims(img, axis=-1)  # Add channel: (48, 48, 1)
        return np.expand_dims(img, axis=0)  # Add batch: (1, 48, 48, 1)

    def predict_frames(self):
        preds = []
        frame_files = sorted(os.listdir(self.frames_folder))

        for frame in frame_files:
            if frame.endswith('.jpg'):
                img_path = os.path.join(self.frames_folder, frame)
                img = self.preprocess_image(img_path)
                prediction = self.model.predict(img, verbose=0)[0]
                class_pred = self.classnames[np.argmax(prediction)]
                preds.append(class_pred)
                print(f'{frame}: {class_pred}')

        return preds
