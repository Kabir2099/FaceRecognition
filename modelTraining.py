
import cv2
import numpy as np
from os import listdir
from os.path import isfile,join

image_path='imageLocation/'
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

onlyFile=[f for f in listdir(image_path) if isfile(join(image_path,f))]
# print(onlyFile)
traning_data,Labels=[],[]

for i,files in enumerate(onlyFile):
    image_loc=image_path+onlyFile[i]
    images=cv2.imread(image_loc,0)

    traning_data.append(np.asarray(images,dtype=np.uint8))
    # print(traning_data)
    # print(type(traning_data))
    Labels.append(i)
    # print(Labels)

Lables=np.asarray(Labels,dtype=np.int32)
print(Labels)

model= cv2.face.LBPHFaceRecognizer_create()
model.train(np.asarray(traning_data),np.asarray(Labels))

print("model training complete")

def face_detector(img,size=0.5):
    roi=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(roi,1.3,5)

    if faces is ():
        return img,[]

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(123,0,23),3)
        roi=img[y:y+h,x:x+w]
        roi=cv2.resize(roi,(400,300))
    return img,roi
        

cap=cv2.VideoCapture(0)

while True:
    ret,frame=cap.read()

    image,face=face_detector(frame)

    try:
        face=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        re=model.predict(face)
        # print(re)

        if re[1]<500:
            confidence=int(100*(1-(re[1])/300))
            # print(confidence)

            display_str=str(confidence)+'% Confidence it is user'
            cv2.putText(image,display_str,(200,200),cv2.FONT_HERSHEY_DUPLEX,1,(51,255,187),4)

            if confidence>=70:
                cv2.putText(image,"Unlocked",(400,400),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,1,(204,34,0),3)
                cv2.imshow("Face System",image)
            else:
                cv2.putText(image,"Locked",(400,400),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,1,(179,0,0),3)
                cv2.imshow("Face System",image)
    
    except:
        cv2.putText(image,"face Not Found",(200,200),cv2.FONT_HERSHEY_COMPLEX,1,(0,89,179),3)
        cv2.putText(image,"Locked",(400,400),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,1,(179,0,0),3)
        cv2.imshow("Face System",image)
    
    if cv2.waitKey(1)==13:
        break

cap.release()
cv2.destroyAllWindows()






    