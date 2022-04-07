import cv2
import numpy as np

face_classfier=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap=cv2.VideoCapture(0)
count=0

def face_extractor(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_classfier.detectMultiScale(gray,1.3,5)

    if faces is ():
        return None
    
    for (x,y,w,h) in faces:
        gray=img[y:y+h,x:x+w]
    
    return gray
        

while True:
    ret,frame=cap.read()
    
    if face_extractor(frame) is not None:
        count+=1
        face=cv2.resize(face_extractor(frame),(400,300))
        face=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        file_name='imageLocation/kabir'+str(count)+'.jpg'
        cv2.imwrite(file_name,face)
        cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),3)
        cv2.imshow("Face apear",face)

    else:
         print("face not found")
    
    if cv2.waitKey(1)==13 or count==100:
        break

cap.release()
cv2.destroyAllWindows()