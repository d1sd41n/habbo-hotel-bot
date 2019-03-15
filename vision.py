import cv2
from mss import mss
from PIL import Image
import numpy as np
import time

def convert_rgb_to_bgr(img):
        return img[:, :, ::-1]

monitor = {'top': 0, 'left': 0, 'width': 1920, 'height': 1080} # write here the resolution of you screen
screen = mss()
frame = None

sct_img = screen.grab(monitor)
img = Image.frombytes('RGB', sct_img.size, sct_img.rgb)
img = np.array(img)
img_bgr = convert_rgb_to_bgr(img)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
#print(img)

habbo_avatar_1 = cv2.imread('assets/habbo_avatar_1.png',0)
w, h = habbo_avatar_1.shape[::-1] # weight height

res = cv2.matchTemplate(img_gray, habbo_avatar_1, cv2.TM_CCOEFF_NORMED)

threshold = 0.9
loc = np.where( res >= threshold)

for pt in zip(*loc[::-1]):
    print(pt)
    print(type(pt))
    cv2.rectangle(img_gray, pt, (pt[0] + w, pt[1] + h), (255,255,255), 4)


chair_1 = cv2.imread('assets/chair1.png',0)
w, h = chair_1.shape[::-1] # weight height

res = cv2.matchTemplate(img_gray, chair_1, cv2.TM_CCOEFF_NORMED)

threshold = 0.5
loc = np.where( res >= threshold)

for pt in zip(*loc[::-1]):
    print(pt)
    print(type(pt))
    cv2.rectangle(img_gray, pt, (pt[0] + w, pt[1] + h), (255,255,255), 4)

cv2.imshow('img_bgr', img_gray)
cv2.imshow('habbo', habbo_avatar_1)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(sct_img)
