import cv2
import glob

class Preprocessor:
    def __init__(self, path_txt):
        self.txt = open(path_txt, 'r') 
        self.lines = self.txt.readlines() 
    
    def split_and_invert(self):
        #h,w = 55,43
        h,w = 100,78
        for i,line in enumerate(self.lines):
            entry = line.strip().split(" ")
            filename = entry[0].split(".")[0]
            im_path = "data/scut/raw/" + entry[0]
            im = cv2.imread(im_path, 0)
            #im = cv2.bitwise_not(im)
            labels = entry[1].split(",")
            for j, label in enumerate(labels):
                offset = w * j
                sub_im = im[0:h, offset:offset + w]
                cv2.imwrite("data/scut/preprocessed/easy_samples/test/{}/{}{}.png".format(label,
                                                i,j), sub_im)
  
if __name__ == "__main__":
    preprocessor = Preprocessor("data/scut/easy_samples_test.txt")
    preprocessor.split_and_invert()