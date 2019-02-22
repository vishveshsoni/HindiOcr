from PIL import Image
import numpy as np
import cv2
import os
import pandas as pd 

img = cv2.imread('D://wowels//2//001_01.jpg')
folder = 'D://wowels//vowels//12'

# i=0
for filename in os.listdir(folder):
    img = cv2.imread(os.path.join(folder,filename))
    if img is not None:
    	re = cv2.resize(img, (32, 32))
    	g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    	arr = np.array(g)
    	im_bw = cv2.bitwise_not(arr)
    	s = im_bw.ravel()
    	df = pd.DataFrame(s).transpose()
    	with open('foo.csv', 'a') as f:
    		df.to_csv(f, index=False,header=False)


# np.savetxt("foo.csv", i, delimiter=",")