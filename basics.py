import cv2
import face_recognition
 
imgE = face_recognition.load_image_file('ImagesBasic/Amitabh Bachan.jpg')
imgE = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)
imgT = face_recognition.load_image_file('ImagesBasic/Anil Kapoor.jpg')
imgT = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)
 
faceLoc = face_recognition.face_locations(imgE)[0]
encodeElon = face_recognition.face_encodings(imgE)[0]
cv2.rectangle(imgE,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)
 
faceLocTest = face_recognition.face_locations(imgT)[0]
encodeTest = face_recognition.face_encodings(imgT)[0]
cv2.rectangle(imgT,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)
 
results = face_recognition.compare_faces([encodeElon],encodeTest)
faceDis = face_recognition.face_distance([encodeElon],encodeTest)
print(results,faceDis)
cv2.putText(imgT,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
 
cv2.imshow('Amitabh Bachan',imgE)
cv2.imshow('amitabh bachchan test',imgT)
cv2.waitKey(0)
