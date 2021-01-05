import models.cnn.letnet as ln
import cv2
import torch
import numpy as np

class recognizer:
    def __init__(self, thresh=0.7):
        self.thresh = thresh

    def find_regions_of_interest(self, image):
        """ 
        Returns the contours around the digits to recognize.
        """

        # apply Gaussian filter for smoothing effect of shadows etc. before adaptive threshold
        # use 5x5 kernel with Sigma = 0, which gives us StD = 0.3*((ksize-1)*0.5 - 1) + 0.8
        blurred = cv2.GaussianBlur(image, (27, 27), 0)

        # use Adaptive threshold to get threshold values based on different regions.
        # use blocksize 7 for region size and 4 as constant to subtract from the Gaussian weighted sum
        binary = cv2.adaptiveThreshold(blurred, 255,  cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 4)

        # erode
        kernel = np.ones((3,3), np.uint8)
        binary = cv2.erode(binary, kernel)
        
        # dilate
        kernel = np.ones((30,30), np.uint8)
        binary = cv2.dilate(binary, kernel)

        cv2.imwrite("bin.png", binary)        
        # find contours
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = []
        rects = np.empty((1,4), np.uint16)
        
        # filter contours
        for i,h in enumerate(hierarchy[0]):
            if h[-1] == -1:
                cnts.append(contours[i])
                poly = cv2.approxPolyDP(contours[i], 3, True)
                rects = np.append(rects, [cv2.boundingRect(poly)], axis=0)

        areas = rects[:,2] * rects[:,3]
        rects = rects[ areas > 0 ]
        areas = areas[ areas > 0 ]
        return rects

    def predict(self, image):
        # sample patches
        rois = self.find_regions_of_interest(image)
        
        # load predictor
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = ln.LetNet5().to(device)
        model.load()

        # get predictions with high confidence
        digits = []
        for i,roi in enumerate(rois):
            x,y,w,h = roi
            roi = image[y:y+h, x:x+w]

            # predict
            pred,conf = model.predict(roi)

            if conf > self.thresh:
                digits.append((i, pred))
                
                # draw results
                cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0))
            
                cv2.putText(image, "id:{}".format(i),
                    (x,y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.3, (0,255,0), 1)
            
                cv2.putText(image, "pred:{}".format(pred),
                    (x,y+10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.3, (0,255,0), 1)

                cv2.putText(image, "conf:{:.2f}".format(conf),
                    (x,y+20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.3, (0,255,0), 1)
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
        #'images/test/cropped_1.png',
        #'images/test/cropped_2.png',
        'images/test/res.png'
    ]

    #model used to predict:
    model = recognizer()


    #run predictions:
    imgs = []
    for pred in pred_list:
        im = cv2.imread(pred, 0)
        imgs.append(im)
        predictions = model.predict(im)
        print(predictions)

    #write input images as concatenated images:
    cv2.imwrite("res.png", util.vconcat_resize_min([img for img in imgs]))
