import models.cnn.letnet as ln
import cv2
import torch
import numpy as np

def vconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    w_min = min(im.shape[1] for im in im_list)
    im_list_resize = [cv2.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation)
                      for im in im_list]
    return cv2.vconcat(im_list_resize)

def ocr_core(filename):
    """
    This function will handle the core OCR processing of images.
    """
    image = cv2.imread(filename, 0)
    #gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # converting image to binary
    #ret, thresh = cv2.threshold(gray_image, 127, 255, 0)
    #thresh = cv2.bitwise_not(thresh)
    digits = []
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = ln.LetNet5().to(device)
    model.load()
 
    #Todo: remove this -> find learned roi 
    #h,w = 55,43 #scut difficult
    h,w = 100,78 #scut easy
    #h,w = 46,42 #black
    for i in range(0,5):
        offset = i * w
        roi = image[0:100, offset:offset+w]

        # predict
        pred = model.predict(roi)
        digits.append(pred)

    print(digits)

pred_list = [
    '/root/ocrm/models/cnn/data/scut/raw/easy_samples/10898.png',
    '/root/ocrm/models/cnn/data/scut/raw/easy_samples/10896.png',
    '/root/ocrm/models/cnn/data/scut/raw/easy_samples/10897.png',
    '/root/ocrm/models/cnn/data/scut/raw/easy_samples/10895.png',
    '/root/ocrm/models/cnn/data/scut/raw/easy_samples/10894.png',
    '/root/ocrm/models/cnn/data/scut/raw/easy_samples/10893.png',
    '/root/ocrm/models/cnn/data/scut/raw/easy_samples/10892.png',
    '/root/ocrm/models/cnn/data/scut/raw/easy_samples/10891.png',
    '/root/ocrm/models/cnn/data/scut/raw/easy_samples/10890.png',
]
imgs = []
for pred in pred_list:
    imgs.append(cv2.imread(pred))
    ocr_core(pred)
cv2.imwrite("res.png", vconcat_resize_min([img for img in imgs]))