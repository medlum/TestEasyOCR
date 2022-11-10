import easyocr
import cv2
from pathlib import Path
from imutils import *
from pathlib import Path
from test_bbox import BboxOCR
from test_ocr import ResultOCR

read_path= Path.cwd()/"unit_img_origin_test"
write_path = "unit_bbox/"
groundtruth = ["063228", "1160", "051041", "0102","212400","16409", "5360", "10189","2359", 
"0709", "0712", "0615", "041645","0807", "11727", "1010", "10656","0231", "111324"]


def main():
    
    BboxOCR(read_path, write_path)
    ResultOCR(read_path, groundtruth)

if __name__ == "__main__":
    main()



