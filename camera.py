import cv2
from model import EmotionsModel
import numpy as np

facec = cv2.CascadeClassifier('/Users/nastyastrashnova/Desktop/prjct/haarcascade_frontalface_default.xml')
model = EmotionsModel("/Users/nastyastrashnova/Desktop/prjct/model.json", "/Users/nastyastrashnova/Desktop/prjct/model/model.h5")
font = cv2.FONT_HERSHEY_SIMPLEX

class VideoCamera(object):
    def __init__(self):
        #self.video = cv2.VideoCapture('/Users/nastyastrashnova/Desktop/Project/videos/facial_exp.mkv')
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self):
        _, fr = self.video.read()
        gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(gray_fr, 1.3, 5) #image, scaleFactor, max faces
        for (x, y, w, h) in faces:
            fc = gray_fr[y:y+h, x:x+w]
            roi = cv2.resize(fc, (48, 48))
            pre = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])

            pred = pre[0] #prediction
            cv2.putText(fr, pred , (x, y), font, 1,(0,0,255), 2)#текста
            list = pre[1]
            a = -150
            for i in range(7):
                a = a+30
                if list[7][i] == pred:
                    colour = (0,0,255)
                else:
                    colour = ( 237,168,62)
                cv2.putText(fr, str(list [7] [i]) + ' ' + str(list [i]) + '%', (x - 230, y - a), font, 1, colour, 2)

            cv2.rectangle(fr, (x,y), (x+w,y+h), (253,166,218), 2) #рамка

        _, jpeg = cv2.imencode('.jpg', fr)
        return jpeg.tobytes()

