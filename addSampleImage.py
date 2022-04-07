import cv2
import numpy as np
import os
face_classfier=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_extractor(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_classfier.detectMultiScale(gray,1.3,5)

    if faces is ():
        return None
    
    for (x,y,w,h) in faces:
        gray=img[y:y+h,x:x+w]
    
    return gray
count=0
li=os.listdir("C:\\Users\\KABIR\\Desktop\\PYTHON\\opencv\\image\\")
i=0
while(i<len(li)):
    img_loc='C:\\Users\\KABIR\\Desktop\\PYTHON\\opencv\\image\\'+li[i]

    img=cv2.imread(img_loc)

    if face_extractor(img) is not None:
        count+=1
        face=cv2.resize(face_extractor(img),(400,300))
        face=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        file_name='imageLocation/name'+str(count)+'.jpg'
        cv2.imwrite(file_name,face)
        # cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),3)
        # cv2.imshow("Face apear",face)

    else:
         print("face not found")
    
    i=i+1


cv2.destroyAllWindows()

    
