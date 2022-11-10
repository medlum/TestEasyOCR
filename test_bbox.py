import easyocr
import cv2
from imutils import *


def BboxOCR(read_path, write_path):
    reader = easyocr.Reader(['en'], gpu=False)
    index = 1
    for image in read_path.glob("*.jpg"):

        print(f"[PROCESSING IMAGE] {image.name}...")
        img = cv2.imread(str(image))
        img = contrast_lab(img,clipLimit = 2.0, tileGridSize=(10,10))

        result = reader.readtext(img, paragraph="False")

        count = 0
        for (bbox, text) in result:
            count += 1
            (tl, tr, br, bl) = bbox
            tl = (int(tl[0]), int(tl[1]))
            tr = (int(tr[0]), int(tr[1]))
            br = (int(br[0]), int(br[1]))
            bl = (int(bl[0]), int(bl[1]))
            cv2.rectangle(img, tl, br, (0,255,0), 2)
            cv2.putText(img, "(" +str(count) + ")" +  " " + text , (tl[0], tl[1]-10),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0),2)

        cv2.imwrite(f"{write_path}bb_unit{index}.png", img)
        index += 1
