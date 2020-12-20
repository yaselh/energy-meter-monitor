import models.cnn.letnet as ln
import cv2
import torch
import numpy as np

class recognizer:
    def __init__(self):
        pass

    def predict(self, image):
        """
        This function will handle the core OCR processing of images.
        """
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

        return digits




class util:
    def vconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
        w_min = min(im.shape[1] for im in im_list)
        im_list_resize = [cv2.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation)
                      for im in im_list]
        return cv2.vconcat(im_list_resize)


if __name__ == "__main__":
    
    #list of image to perform predition on:
    pred_list = [
        'images/test/cropped_1.png',
        'images/test/cropped_2.png'
    ]

    #model used to predict:
    model = recognizer()

    imgs = []
    for pred in pred_list:
        im = cv2.imread(pred, 0)
        imgs.append(im)
        predictions = model.predict(im)
        print(predictions)
    cv2.imwrite("res.png", util.vconcat_resize_min([img for img in imgs]))
