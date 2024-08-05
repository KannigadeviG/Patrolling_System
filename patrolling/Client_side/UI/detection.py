from PyQt5.QtWidgets import QMainWindow
from PyQt5.uic import loadUi
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
import cv2
import numpy as np
import time
import requests

class Detection(QThread):
    def __init__(self,token,location,receiver):
        super(Detection,self).__init__()
        self.token=token
        self.location=location
        self.receiver=receiver

    # Define the changePixmap signal
    changePixmap = pyqtSignal(QImage)

    def run(self):
        self.running=True
        net = cv2.dnn.readNet("weights/yolov4-obj_last.weights", "cfg/yolov4-obj.cfg")
        classes = []
        with open("obj.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]
        
        layer_names = net.getLayerNames()
        output_layers = []
        unconnected_layers = net.getUnconnectedOutLayers()

        if isinstance(unconnected_layers, int):
            output_layers.append(layer_names[unconnected_layers - 1])
        elif isinstance(unconnected_layers, list):
            for layer_index in unconnected_layers:
                output_layers.append(layer_names[layer_index - 1])
        colors = np.random.uniform(0, 255, size=(len(classes), 3))
        font = cv2.FONT_HERSHEY_PLAIN
        starting_time = time.time() - 11

        cap = cv2.VideoCapture(0)
        
        while self.running:
            ret, frame = cap.read()
            if ret:
                height, width, channels = frame.shape
                blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
                net.setInput(blob)
                outs = net.forward(output_layers)
                class_ids = []
                confidences = []    
                boxes = []
                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]

                        if confidence > 0.5:  # Adjusted confidence threshold
                            center_x = int(detection[0] * width)
                            center_y = int(detection[1] * height)
                            w = int(detection[2] * width)
                            h = int(detection[3] * height)
                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)

                            boxes.append([x, y, w, h])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)

                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)  # Adjusted NMS threshold

                for i in range(len(boxes)):
                    if i in indexes:
                        x, y, w, h = boxes[i]
                        label = str(classes[class_ids[i]])
                        confidence = confidences[i]
                        color = (0, 0, 255)  # Changed color to red (BGR format)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(frame, f'{label} {confidence:.1%}', (x, y - 20), font, 1, color, 1)
                        elapsed_time = starting_time - time.time()

                        if elapsed_time <= -10:
                            starting_time = time.time()
                            self.save_detection(frame)

                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                bytesPerLine = channels * width
                convertToQtFormat = QImage(rgbImage.data, width, height, bytesPerLine, QImage.Format_RGB888)
                p = convertToQtFormat.scaled(854, 480, Qt.KeepAspectRatio)
                
                # Emit the changePixmap signal with the QPixmap object
                self.changePixmap.emit(p)

    def save_detection(self, frame):
        cv2.imwrite("saved_frame/frame.jpg", frame)
        print('Frame Saved')
        self.post_detection()
    def post_detection(self):
        try:
            url = 'http://127.0.0.1:8000/api/images/'
            headers = {'Authorization': 'Token ' + self.token}
            files = {'image': open('saved_frame/frame.jpg', 'rb')}
            data = {'user_ID': self.token,'location': self.location, 'alert_receiver': self.receiver}
            response = requests.post(url, files=files, headers=headers, data=data)

			# HTTP 200
            if response.ok:
                print('Alert was sent to the server')
			# Bad response
            else:
                print('Unable to send alert to the server')
				
        except:
            print('Unable to access server')  


# You might need to instantiate and start the Detection thread in your QMainWindow.
