import easyocr
import cv2
from pathlib import Path
from imutils import *

label = ["063228", "1160", "051041", "0102","212400","16409", "5360", "10189","2359", 
"0709", "0712", "0615", "041645","0807", "11727", "1010", "10656","0231", "111324"]

def ResultOCR(read_path, groundtruth):
    reader = easyocr.Reader(['en'], gpu=False)

    predict = []
    score = 0
    unit_num = []
    img_num = []
    index = 0

    for image in read_path.glob("*.jpg"):

        img = cv2.imread(str(image))
        img = contrast_lab(img,clipLimit = 2.0, tileGridSize=(10,10))

        result = reader.readtext(img, paragraph="False")

        if len(result) != 0:

            unit = str(result[0][1])
            u = [i for i in unit if i.isdigit()]
            u = "".join(u)
            predict.append(u)
                
        else: 
            u = "nil"
            predict.append("failure")    

        if u == label[index]:
            print(f"[IMAGE: {image.name}] UNIT LABEL : {label[index]} | UNIT OCR : {u} | MATCHED")
        else: 
            print(f"[IMAGE: {image.name}] UNIT LABEL : {label[index]} | UNIT OCR : {u} | INCORRECT")

        index += 1

    for i in predict:
            if i in groundtruth:
                score += 1

    accuracy_score = round(score/len(groundtruth),2)

    print(f"TOTAL MATCHED: {score} out of {len(groundtruth)} or {accuracy_score * 100}%")



