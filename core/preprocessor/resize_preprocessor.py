import cv2

class ResizePreprocessor():
    
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.inter = inter
        
    def preprocess(self):
        return cv2.resize((self.width, self.height), interpolation=self.inter)
