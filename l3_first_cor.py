import cv2
import pytesseract

img = cv2.imread("./images/py.jpg")


#! Add the following config, if you have tessdata error like: "Error opening data file..."
# # Example config: r'--tessdata-dir "C:\Program Files (x86)\Tesseract-OCR\tessdata"'
# It's important to add double quotes around the dir path.
# tessdata_dir_config = r'--tessdata-dir "<replace_with_your_tessdata_dir_path>"'


#! --psm Page segmentation method
# 0    Orientation and script detection(OSD) only.
# 1    Automatic page segmentation with OSD.
# 2    Automatic page segmentation, but no OSD, or OCR.
# 3    Fully automatic page segmentation, but no OSD. (Default)
# 4    Assume a single column of text of variable sizes.
# 5    Assume a single uniform block of vertically aligned text.
# 6    Assume a single uniform block of text.
# 7    Treat the image as a single text line.
# 8    Treat the image as a single word.
# 9    Treat the image as a single word in a circle.
# 10   Treat the image as a single character.
# 11   Sparse text. Find as much text as possible in no particular order.
# 12   Sparse text with OSD.
# 13   Raw line. Treat the image as a single text line, bypassing hacks that are Tesseract-specific.


#! --oem  Optical Engine Mode
# 0. Legacy engine only.
# 1. Neural nets LSTM engine only.
# 2. Legacy + LSTM engines.
# 3. Default, based on what is available.

# Adding custom options
custom_config = r'--oem 3 --psm 4'
text = pytesseract.image_to_string(img, config=custom_config)
print(text)
