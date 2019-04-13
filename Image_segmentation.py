import cv2
import numpy as np
import sys
import os

def get_rect_rank(rect):
    x_mean=(rect[0]+rect[2])/2
    y_mean=(rect[1]+rect[3])/2
    rank = (y_mean/50)*50000+x_mean
    return rank
    
path = "h4.jpg"
img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
feat_file = open("character.feat",'w')
img[img>=128]=255
img[img<128]=0

img_area=img.shape[0]*img.shape[1]

character_height=[]
above_character_height=[]
below_character_height=[]
character_width=[]
stroke_thickness=[]
    
out_img =img.copy()
contour_img= img.copy()

img = 255 - img

contours, hierarchy = cv2.findContours(img,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
contour_rects = []
#out_img = cv2.drawContours(out_img, contours, -1, (0,0,0), 3)
k=0
cnt=0
for (i, j) in zip(contours, hierarchy[0]):
    if cv2.contourArea(i) > img_area/150000 and j[3] == -1:
        """ Minimum Area Rectangle
        rect = cv2.minAreaRect(i)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        out_img = cv2.drawContours(out_img,[box],0,(0,0,255),2)
        """
        
        x,y,w,h = cv2.boundingRect(i)
        contour_rects.append([x,y,x+w,y+h])

contour_rects.sort(key=lambda x:get_rect_rank(x))
for x,y,x_w,y_h in contour_rects:
        #cv2.imwrite('word_' + str(k).zfill(3) + '.jpg',out_img[y:y+h,x:x+w])
        contour_img = cv2.rectangle(contour_img,(x,y),(x_w,y_h),(0,0,0),2)
        k += 1
        # cv2.putText(contour_img,str(k),(x,y),cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
 
cv2.imwrite("contours.jpg",contour_img)
print len(contour_rects)
'''
Deskewing starts here
'''
for x,y,x_w,y_h in contour_rects:
    print ("1","vish")
    word = out_img[y:y_h,x:x_w].copy()
    word = 255 - word
     
    minLineLength = word.shape[1]
    maxLineGap = 20
    #longest_line_index=0
    longest_line=[0,0]
    confiedence = 100
    lines = None
    deskewed = None

    while(lines is None):
        #lines = cv2.HoughLinesP(word,1,np.pi/180,confiedence,minLineLength,maxLineGap)
        lines = cv2.HoughLinesP(word,rho = 1,theta = 1*np.pi/180,threshold = 100,minLineLength = minLineLength,maxLineGap = 20)
        #print len(lines)
        confiedence -= 5
        if(confiedence == 50):
            break
    
    if(not(lines is None)):
        for loc,line in enumerate(lines):
            for x1,y1,x2,y2 in line:
                dist = np.sqrt(abs((x2-x1)^2-(y2-y1)^2))
                angle = np.arctan((y2-y1)/float(x2-x1)) * 180.0/np.pi
                if(dist > longest_line[0] and angle < 30 and angle > -30):
                    longest_line[0] = dist
                    longest_line[1] = angle
                    #longest_line_index = loc
        #x1,y1,x2,y2=lines[longest_line_index][0]
        #angle = np.arctan((y2-y1)/float(x2-x1)) * 180.0/np.pi

        rows,cols = word.shape
        rot = cv2.getRotationMatrix2D((cols/2,rows/2),longest_line[1],1)
        rotated = 255 - cv2.warpAffine(word,rot,(cols,rows),cv2.INTER_CUBIC)

        blur = cv2.GaussianBlur(rotated,(5,5),0)
        ret3,deskewed = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    else:
        print("2")
        k += 1
        cv2.imwrite("word"+"_"+str(k).zfill(3)+".jpg",255-word)
        ret3,deskewed = cv2.threshold(255-word,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    if(word.shape[1]<img.shape[1]/60):
        k += 1
        cv2.imwrite("word"+"_"+str(k).zfill(3)+".jpg",255-word)
        ret3,deskewed = cv2.threshold(255-word,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    
    print("3")
    

    ''' 
    #char splitting starts here
    '''

    horizontal_histogram = (255*deskewed.shape[1])-deskewed.sum(axis=1)

    #plt.plot(horizontal_histogram,'ro')
    #plt.show()

    upper_start=-1
    lower_start=-1
    find_upper=True

    for loc,i in enumerate(horizontal_histogram):
        if( i/255 > deskewed.shape[1]/1.5 and upper_start==-1 and find_upper and horizontal_histogram[loc]<=horizontal_histogram[loc+1] and horizontal_histogram[loc+1] >= horizontal_histogram[loc+2]):
            upper_start=loc+1
            #print upper_start
        if( upper_start!=-1 and find_upper and i/255 < deskewed.shape[1]/1.5):
            '''and horizontal_histogram[loc]>=horizontal_histogram[loc+1] and horizontal_histogram[loc+1] <= horizontal_histogram[loc+2]'''
            upper_end = loc+1
            #print upper_end
            find_upper = False
        #and horizontal_histogram[loc]>=horizontal_histogram[loc+1] and horizontal_histogram[loc+1] <= horizontal_histogram[loc+2]
        if(i/255 < deskewed.shape[1]/4 and loc > deskewed.shape[0]*3/4 and lower_start==-1 and not find_upper ):
            lower_start=loc+1
            break
    #lower_start=horizontal_histogram[upper_end:].argmin()
    stroke_width=upper_end-upper_start
    stroke_thickness.append(upper_end-upper_start)
    cv2.imwrite(path.replace(".jpg","")+"_"+str(k)+"_upper.jpg",deskewed[:upper_start])
    cv2.imwrite(path.replace(".jpg","")+"_"+str(k)+"_lower.jpg",deskewed[lower_start:])
    cv2.imwrite(path.replace(".jpg","")+"_"+str(k)+"_middle.jpg",deskewed[upper_end:lower_start])
    upper = deskewed[:upper_end]
    middle = deskewed[upper_end:lower_start]
    lower = deskewed[lower_start:]
    
    vertical_histogram = (255*middle.shape[0])-middle.sum(axis=0)

    character_height.append(lower_start-upper_end if lower_start!=-1 else -1)
    above_character_height.append(upper_start)
    below_character_height.append(deskewed.shape[0]-lower_start)

    vertical_seg=[]
    vertical_break=[]
    
    '''
    percentile=[]

    for i in range(0,50,2):
            percentile.append(np.percentile(vertical_histogram,i))
    '''

    flag=False
    #for loc,data in enumerate(vertical_histogram < np.percentile(vertical_histogram,np.argmax(np.gradient(percentile))*2)):
    for loc,data in enumerate(vertical_histogram > 255*stroke_width/2):
            if(data):
                    if(not flag):
                        vertical_break.append(loc)
                        flag=True
            else:
                    if(flag):
                        vertical_break.append(loc)
                        flag=False

    print vertical_break,data
    '''
    for loc,data in enumerate(vertical_histogram):
        if(data/255 < stroke_width):
            vertical_seg.append(loc)
    for i in range(len(vertical_seg)-1):
        if(vertical_seg[i+1] - vertical_seg[i] > 3):
            vertical_break.append(vertical_seg[i])
            vertical_break.append(vertical_seg[i+1])
    #print vertical_break
    '''
    spacer_array=np.zeros((deskewed.shape[0],2))
    spacer_array += 255
    spacer_array[:,1]=0
    output = np.zeros((deskewed.shape[0],1))
    # k += 1
    #print k
    j=1
    feat_file.write("word"+"_"+str(k).zfill(3)+"\n\n")
    for i in range(0,len(vertical_break)-1,2):
        print ("1","------------------")
        character_width.append((k,vertical_break[i+1]-vertical_break[i]))
        character=deskewed[:,vertical_break[i]:vertical_break[i+1]]
        resized = cv2.resize(character,(9,12),cv2.INTER_AREA)
        # resized_image = cv2.resize(character, (50, 50)) 
        cv2.imwrite("ch_"+str(k)+"_"+str(j)+".jpg",character)
        j+=1

        output=np.concatenate((output,character,spacer_array),axis=1)
        #step = np.around(np.sqrt(character.shape/25))
        #character=np.concatenate((character,np.zeros((character.shape[0],character.shape[1]%step))),axis=1)

        #resized[resized>64]=0
        #resized[resized<=64]=1
        #vertical_feat = ((255*resized.shape[0])-resized.sum(axis=0))/255
        #horizontal_feat = ((255*resized.shape[1])-resized.sum(axis=1))/255
        feat_file.write("character: "+str(i/2).zfill(2)+" "+str(resized.shape[0])+"x"+str(resized.shape[1])+"\n")
        for col in resized:
            for row in col:
                feat_file.write("0" if row < 64 else " ")
            feat_file.write("\n")
        feat_file.write("\n\n")
        cv2.imwrite("word"+"_"+str(k).zfill(3)+".jpg",output)    
    #cv2.imwrite(sys.argv[2],deskewed)
    #plt.plot(vertical_histogram)
    #plt.show()