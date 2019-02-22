import cv2
from keras.models import load_model
import numpy as np
from collections import deque

# model2 = load_model('devanagari_refined_voewl.h5')
model1 = load_model('devanagari_refined_final.h5')
model3 = load_model('devanagari_refined_new.h5')
model4 = load_model('devanagari_refined_final_eight.h5')

# model1 = load_model('D:\ewocr\devanagari-character-recognition\model.h5')





def main():
    letter_count = {0: 'CHECK', 1: 'क', 2: 'ख', 3: 'ग', 4: 'घ', 5: 'ङ', 6: 'च',
                    7: 'छ', 8: 'ज', 9: 'झ', 10: 'ञ',
                    11: 'ट',
                    12: 'ठ', 13: 'ड', 14: 'ढ', 15: 'ण', 16: 'त', 17: 'थ',
                    18: 'द',

                    19: 'ध', 20: 'न', 21: 'प', 22: 'फ',
                    23: 'ब',
                    24: 'भ', 25: 'म', 26: 'य', 27: 'र', 28: 'ल', 29: 'व', 30: 'श',
                    31: 'ष',32: 'स', 33: 'ह',
                    34: 'क्ष', 35: 'त्र', 36: 'ज्ञ',
                    37: '0', 38: '1',39: '2',40: '3',41:'4',42:'5',43:'6',44:'7',45:'8',46:'9',
                    47: 'अ',48: 'आ ',49: 'इ ',50: 'ई',51: 'उ',52: 'ऊ',53: 'ए',
                    54: 'ऐ',55: 'ओ',56: 'औ',57:'अं',58:'अः',
                    59: 'CHECK'
                    }


    # for i in range(0,5):

    #     sa = str(i)+".png"
    #     path ="D:/wowels/let/"+ sa
    
    #     large = cv2.imread(path,0)
    #     img = cv2.resize(large, (32, 32))


    #     # gray_image = cv2.cvtColor(large,cv2.COLOR_BGR2GRAY)
    #     im_bw = cv2.bitwise_not(large)
    #     # (thresh, im_bw) = cv2.threshold(large, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #     (thresh, im_bw) = cv2.threshold(im_bw, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # # 
    # # thresh = 127
    # # im_bw = cv2.threshold(gray_image, thresh, 255, cv2.THRESH_BINARY)[1]
    
    
    #     cv2.imshow('exp',large)
    #     cv2.waitKey(0)
    #     classs , prob = keras_predict(model3, im_bw)
    #     d = str(letter_count[classs])
    #     print(d)
    #     print(prob)

    sa ="D:/digi.png"
    large = cv2.imread(sa)
    larges = cv2.resize(large, (32, 32))
    gray_image = cv2.cvtColor(larges,cv2.COLOR_BGR2GRAY)
    im_bw = cv2.bitwise_not(gray_image)
    # (thresh, im_bw) = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    (thresh, im_bw) = cv2.threshold(im_bw, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    

    cv2.imshow('exp',large)
    cv2.waitKey(0)
    classs , prob = keras_predict(model4, im_bw)
    d = str(letter_count[classs])
    print(d)
    print(prob)
    print(classs)

def keras_predict(model, image):
    processed = keras_process_image(image)
    # print("processed: " + str(processed.shape))

    pred_probab = model.predict(processed)[0]
    pred_class = list(pred_probab).index(max(pred_probab))
    
    # print(max(pred_probab))
    return pred_class , max(pred_probab)


def keras_process_image(img):
    image_x = 32
    image_y = 32
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (-1, image_x, image_y, 1))
    return img


main()
