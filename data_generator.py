import cv2

from keras.utils.np_utils import to_categorical
import numpy as np


class ImageDataGenerator(object):
    def __init__(self, rescale=None):
        self.rescale = rescale
        self.reset()

    def reset(self):
        self.images = []
        self.labels = []

    def flow_from_directory(self, img_path, classes, batch_size=32):
        while True:
            for path, cls in zip(img_path, classes):
                img = cv2.imread(path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                self.images.append(np.asarray(img) / 255)
                self.labels.append(cls)

                if len(self.images) == batch_size:
                    inputs = np.asarray(self.images, dtype=np.float32)
                    targets = np.asarray(self.labels, dtype=np.float32)
                    self.reset()
                    yield inputs, targets