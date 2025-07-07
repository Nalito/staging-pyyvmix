import os
import random

class FramePredictor:
    def __init__(self, frames_folder):
        self.frames_folder = frames_folder
        # Seven basic emotions
        self.classnames = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    def predict_frames(self):
        """
        Instead of running a CNN, randomly assign an emotion to each frame.
        """
        preds = []
        frame_files = sorted(os.listdir(self.frames_folder))

        for frame in frame_files:
            if frame.endswith('.jpg'):
                emotion = random.choice(self.classnames)
                preds.append(emotion)
                print(f"{frame}: {emotion}")

        return preds
