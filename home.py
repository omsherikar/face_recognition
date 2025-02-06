import face_recognition
import cv2
import numpy as np 
# from PIL import Image
import csv
from datetime import datetime

video_capture= cv2.VideoCapture(0)

#load known faces 
jyoti_image = face_recognition.load_image_file("faces/jyoti.jpg")
jyoti_encoding = face_recognition.face_encodings(jyoti_image)[0]

om_image = face_recognition.load_image_file("faces/om.jpg")
om_encoding = face_recognition.face_encodings(om_image)[0]

# ayush_image = face_recognition.load_image_file("faces/ayush.jpg")
# ayush_encoding = face_recognition.face_encodings(ayush_image)[0]

shravya_image = face_recognition.load_image_file("faces/shravya.jpg")
shravya_encoding = face_recognition.face_encodings(shravya_image)[0]



# lata_image = face_recognition.load_image_file("faces/lata.jpg")
# lata_encoding = face_recognition.face_encodings(lata_image)[0]

# ramesh_image = face_recognition.load_image_file("faces/ramesh.jpg")
# ramesh_encoding = face_recognition.face_encodings(ramesh_image)[0]

santosh_image = face_recognition.load_image_file("faces/santosh.jpg")
santosh_encoding = face_recognition.face_encodings(santosh_image)[0]

known_face_encodings = [om_encoding,shravya_encoding,jyoti_encoding,santosh_encoding]
known_face_names = ["Om"," Ayush", "Shravya", "Lata"," Ramesh", "Jyoti", "Santosh"]


students = known_face_names.copy()

face_locations = []
face_encodings = []

# get the correct date and time
now = datetime.now()
current_date = now.strftime("%y-%m-%d")
f = open(f"{current_date}.csv", "w+" ,newline="" )

lnwriter = csv.writer(f)
from PIL import Image






while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame,(0,0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    #Recognize face
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distance)


        if(matches[best_match_index]):
            name = known_face_names[best_match_index]


        # Add the text if a person is present  
        if name is known_face_names:
            font=cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (10,100)
            fontScale = 1.5
            fontColor = (255,0,0)
            thickness = 3
            lineType = 2
            cv2.putText(frame, name + "Present", bottomLeftCornerOfText,font,fontScale, fontColor, thickness, lineType)

            if name in students:
                students.remove(name)
                current_time = now.strftime("%H-%M")
                lnwriter.writerow([name, current_time])

    cv2.imshow("Attendence", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


video_capture.release()
cv2.destroyAllWindows()
f.close()

