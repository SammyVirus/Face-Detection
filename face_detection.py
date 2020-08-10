import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eyes_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
video = cv2.VideoCapture(0)
while True:
    check, frame = video.read()
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grayFrame, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)
        roi_grayFrame = grayFrame[y:y+h, x:x+w]
        roi_frame = frame[y:y+h, x:x+w]
        eyes = eyes_cascade.detectMultiScale(roi_grayFrame)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_frame, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            

    cv2.imshow("Face detection", frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        cv2.imwrite('Detected Face and Eyes.jpg', frame)
        break

video.release()
cv2.destroyAllWindows()
