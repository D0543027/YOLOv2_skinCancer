from darkflow.net.build import TFNet
import cv2
import os
import numpy as np

def draw_boundingBox(original_img, predictions):
    newImage = np.copy(original_img)
    #print(predictions)
    for result in predictions:
        top_x = result['topleft']['x']
        top_y = result['topleft']['y']

        btm_x = result['bottomright']['x']
        btm_y = result['bottomright']['y']

        label = result['label']
        confidence = result['confidence']
        if confidence > 0.3:
            if label == "benign":
                newImage = cv2.rectangle(newImage, (top_x, top_y), (btm_x, btm_y), (255,0,0), 1)
                newImage = cv2.putText(newImage, label, (top_x, top_y-5), cv2.FONT_HERSHEY_DUPLEX , 0.5, (0, 230, 0), 1, cv2.LINE_AA)
            else:
                newImage = cv2.rectangle(newImage, (top_x, top_y), (btm_x, btm_y), (220,0,255), 1)
                newImage = cv2.putText(newImage, label, (top_x, top_y-5), cv2.FONT_HERSHEY_DUPLEX , 0.5, (0, 0, 230), 1, cv2.LINE_AA)
    return newImage

if __name__ == "__main__":
    options = {"model": "cfg/tiny-yolov2-custom.cfg", "load": -1,"threshold": 0.2,"gpu":0.7}

    tfnet = TFNet(options)

    img_path = "data/testing"
    output_path = "data/result"


    for roots, dirs, files in os.walk(img_path):
        for fname in (x for x in files if x.endswith(".jpg")):
            fpath = os.path.join(roots,fname)
            img = cv2.imread(fpath)
            results = tfnet.return_predict(img)
            predicted_img = draw_boundingBox(img,results)
            cv2.imwrite(os.path.join(output_path,fname),predicted_img)