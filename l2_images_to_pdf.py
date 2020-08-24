from l1_Warp_Perspective import make_page_upright
from tkinter import Tk, filedialog
from PIL import Image
import cv2
import os
import sys
import webbrowser


PDF_PATH = '../x.pdf'


def get_images_folder():
    Tk().withdraw()
    folder = filedialog.askdirectory()
    if folder == '':
        sys.exit()
    return folder


def create_pil_image(src):
    img = cv2.cvtColor(make_page_upright(src), cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)


folder = get_images_folder()
os.chdir(folder)

pil_images = [create_pil_image(img) for img in os.listdir(folder)]


#! PDFمكتبة قوية جدا مخصصة لملفات ال
#! https://www.blog.pythonlibrary.org/2010/03/08/a-simple-step-by-step-reportlab-tutorial/

# ? صورة واحدة
# p_img.save(PDF_PATH, "PDF", resolution=10)

# ? اكتر من صورة
pil_images[0].save(PDF_PATH, "PDF", resolution=100.0,
                   save_all=True, append_images=pil_images[1:])


webbrowser.open(PDF_PATH)
