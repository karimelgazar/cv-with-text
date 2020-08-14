import cv2
import numpy as np

# ? نفس الدالة من كورس المبتدئين


def resize(image, width=None, height=None, show=False):
    """
        Arguments:
            image {np.array} -- الصورة المراد تغيير حجمها

        Keyword Arguments:
            width {[int]} -- العرض الجديد للصورة بعد تغيير حجمها
            height {[int]} -- الارتفاع الجديد للصورة بعد تغيير حجمها
            show {bool} -- إظهار الصورة التي تم تغيير حجمها أم لا

        Returns:
            [np.array] -- الصورة التي تم تغيير حجمها
    """
    if width is None and height is None:
        return image

    if width is None:
        r = height / image.shape[0]
        width = int(r * image.shape[1])
    elif height is None:
        r = width / image.shape[1]
        height = int(r * image.shape[0])

    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    if show:
        cv2.imshow('Image Resized By {0:.2f}'.format(r), resized)
        cv2.waitKey(0)

    return resized


def draw_corners(img, points):
    """
    دالة بسيطة لرسم النقط المعطاة على الصورة المعطاة
    """
    # ? تم رسم النقط بتلك الالوان المحددة حتى يتم
    # ? التفريق بين الأركان الاربعة المرتبة والغير مرتبة
    if len(points) == 4:
        p1, p2, p3, p4 = points
        cv2.circle(img, tuple(p1), 6, (0, 0, 255), -1)
        cv2.circle(img, tuple(p2), 6, (0, 255, 0), -1)
        cv2.circle(img, tuple(p3), 6, (255, 0, 0), -1)
        cv2.circle(img, tuple(p4), 6, (0, 255, 255), -1)

    else:
        from random import sample
        for p in points:
            random_color = tuple(sample(range(0, 255), 3))
            cv2.circle(img, tuple(p), 6, random_color, -1)

    cv2.imshow("image", img)
    cv2.waitKey(0)


def get_corner_points(contour):
    """
        cv2.arclength(contour, closed)
        ------------
            الدالة بتحسب المحيط 
            closed: [True OR False] هل الشكل مغلق 

        cv2.approxPolyDP(contour, epsilon, closed)
        --------------------------------------------
            contour بترجع اقل عدد من النقط اللى ممكن تمثل اركان ال 
            بيحيث لما بنحسب الأركان بنجيب المحيط اللى بتكونه النقط دى 
            contour وبنقارنه بالمحيط الاصلى بتاع ال 
            contour عادة نسبة الخطا دى عبارة عن (%1 : %5) من المحيط الاصلى بتاع ال 

            epsilon: وهو عبارة عن نسبة الخطأ المسموح به
            closed: [True OR False] هل الشكل مغلق 
    """
    # المحيط
    peri = cv2.arcLength(contour, True)
    # الاركان
    corners = cv2.approxPolyDP(contour, 0.02 * peri, True)

    # ?  هذه حماية اضافية من حدوث خطا فى تحديد الورقة
    # ?  لانه من المعروف ان الورق عبارة عن مستطيل ولها 4 أركان
    # ?  لذا اذا وجد اكثر من ذلك  لم يكن هذا الشكل ورقة
    #! ولــــــــــكـــــــــــــــن
    # ? فى كثير من الأحيان ستكون الورقة مثنية
    # ? ولذلك سيكون لها اكثر من 4 أركان
    # ? لذلك سترجع هذه الدالة جميع النقط المستخرجة
    # ? وفى دالة اخرى سنعالج تلك المشكلة

    # if len(corners) != 4:
    #     return None

    """
        np.squeeze() فائدة
        
        ! قبل 
        [
            [[110  71]]
            [[101 202]]
            [[227 203]]
            [[214  76]]
        ] 
        shape: (4, 1, 2)
        =============================
        ! بعد
        [
            [110  71]
            [101 202]
            [227 203]
            [214  76]
        ]
        shape: (4, 2)
    """

    return np.squeeze(corners)


def order_corner_points_clockwise(points):
    # Initzie قائمة بالإحداثيات التي سيتم طلبها
    # بحيث يكون الإدخال الأول في القائمة أعلى اليمين ،
    # الإدخال الثاني هو أعلى اليمين ، والثالث هو
    # أسفل اليمين ، والرابع هو أسفل اليسار
    rect = np.zeros((4, 2), dtype="float32")

    #! مهم جدا
    #! ==========
    # ? numpy array  الصورة عبارة عن
    # ? العمود الاول من الشمال هو قيم المحور الرأسى
    # ? العمود الثانى من الشمال هو قيم المحور الأفقى

    # ستحتوي النقطة العلوية اليسرى على أصغر جمع ، في حين
    # ستحتوي النقطة السفلية اليمنى على أكبر جمع
    axis_sum = np.sum(points, axis=1)
    rect[0] = points[np.argmin(axis_sum)]
    rect[2] = points[np.argmax(axis_sum)]

    # الآن ، احسب الفرق بين النقاط
    # أعلى يمين النقطة سيكون لها أصغر فرق ،
    # بينما سيكون أسفل اليسار أكبر فرق
    axis_diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(axis_diff)]
    rect[3] = points[np.argmax(axis_diff)]

    # الاركان الاربعة مترتبة
    return rect


def apply_top_view(image, pts):
    (tl, tr, br, bl) = pts

    # حساب عرض الصورة الجديدة التي ستكون
    # أقصى مسافة بين أسفل اليمين وأسفل اليسار
    # x-منسق أو إحداثيات س أعلى يمين وأعلى يسار
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # حساب ارتفاع الصورة الجديدة التي ستكون
    # أقصى مسافة بين أعلى اليمين وأسفل اليمين
    # إحداثيات ص أو إحداثيات ص أعلى اليسار وأسفل اليسار
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # الآن لدينا أبعاد الصورة الجديدة ، قم بالبناء
    # مجموعة نقاط الوجهة للحصول على "منظر عين الطيور" ،
    # (أي عرض من أعلى لأسفل) للصورة ، وتحديد النقاط مرة أخرى
    # في أعلى اليسار ، أعلى اليمين ، وأسفل اليمين ، وأسفل اليسار الترتيب
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # حساب مصفوفة تحويل المنظور ثم تطبيقه
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # إعادة الصورة المشوهة
    return warped


#!==============
# ? الخطوة رقم 0
# ? قراءة الصورة
#!==============
image = cv2.imread("./images/page.jpg")

original_height, original_width = image.shape[:2]
new_height = 400
cv2.imshow("image", image)
cv2.waitKey(0)

# ? خلى الطول 400 بكسل والقيمة دى هنحتاجها فيما بعد
ratio = original_height / new_height

#!==============
# ? الخطوة رقم 1
# ? عملنا تصغير للصورة عشان تبقى العمليات اسرع + عمليات أساسية
#!==============
small_image = resize(image, height=new_height)
cv2.imshow("image", small_image)
cv2.waitKey(0)

gray_small_image = cv2.cvtColor(small_image, cv2.COLOR_BGR2GRAY)
cv2.imshow("image", gray_small_image)
cv2.waitKey(0)

blurred_gray_small_image = cv2.GaussianBlur(gray_small_image, (9, 9), 0)
cv2.imshow("image", blurred_gray_small_image)
cv2.waitKey(0)

canny_blurred_gray_small_image = cv2.Canny(blurred_gray_small_image, 120, 240)
cv2.imshow("image", canny_blurred_gray_small_image)
cv2.waitKey(0)

#!==============
# ? الخطوة رقم 2
# ? تحديد حدود الورقة
#!==============
copy = small_image.copy()
cnts = cv2.findContours(canny_blurred_gray_small_image,
                        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

biggest_contour = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
cv2.drawContours(copy, [biggest_contour], -1, (0, 255, 0), 2)
cv2.imshow("image", copy)
cv2.waitKey(0)


#!==============
# ? الخطوة رقم 3
# ? تحديد نقط اركان الورقة
#!==============
unordered_corners = get_corner_points(biggest_contour)
print(len(unordered_corners))
draw_corners(copy.copy(), unordered_corners)

#! خطوة مهمة جدا
corners = order_corner_points_clockwise(unordered_corners)
draw_corners(copy, corners)
#!==============
# ? الخطوة رقم 4
# ? تعديل منظر الورقة
#!==============
new_image = apply_top_view(image, np.float32(corners)*ratio)
cv2.imshow("image", resize(new_image, height=400))
cv2.waitKey(0)

#!==============
# ? الخطوة رقم 5
# ? تدريبات
#!==============
# TODO جعل الكلام بارز اكثر
# TODO Trackbars استخدم
# TODO السماح للمستخدم بادخال اكثر من صورة للبرنامج
# TODO حفظ الصور فى مجلد
