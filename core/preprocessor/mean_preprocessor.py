import numpy as np
import cv2

class MeanPreprocessor():
    
    def __init__(self, r_mean, g_mean, b_mean):
        self.r_mean = r_mean
        self.g_mean = g_mean
        self.b_mean = b_mean
        
    def preprocess(self, image):
        B, G, R = cv2.split(image.astype("float32"))
        R -= self.r_mean
        G -= self.g_mean
        B -= self.b_mean
        
        image = cv2.merge([B, G, R])
        return image
