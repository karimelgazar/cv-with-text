import cv2
import pytesseract

img = cv2.imread("./images/py.jpg")


#! Add the following config, if you have tessdata error like: "Error opening data file..."
# # Example config: r'--tessdata-dir "C:\Program Files (x86)\Tesseract-OCR\tessdata"'
# It's important to add double quotes around the dir path.
# tessdata_dir_config = r'--tessdata-dir "<replace_with_your_tessdata_dir_path>"'

# Adding custom options
custom_config = r'--oem 3 --psm 4 --tessdata-dir "C:\Program Files (x86)\Tesseract-OCR\tessdata"'
text = pytesseract.image_to_string(img, config=custom_config)
print(text)
