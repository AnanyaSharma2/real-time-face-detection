import cv2

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
video_capture = cv2.VideoCapture(0)
def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    # used to detect faces in the grayscale image
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        # to show green color box
        #cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
        # to show blue color box
        #cv2.rectangle(vid, (x, y), (x + w, y + h), ( 255, 0,0), 4)
        # to show red color box
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 0,255), 4)
    return faces
while True:
    # read frames from the video
    result, video_frame = video_capture.read()  
    # terminate the loop if the frame is not read successfully
    if result is False:
        break  
    # apply the function we created to the video frame
    faces = detect_bounding_box(
        video_frame
    )  
    # display the processed frame in a window named "My Face Detection Project"
    cv2.imshow(
        "My Face Detection Project", video_frame
    )  
    # stop the loop when the '1' key is pressed
    if cv2.waitKey(1) & 0xFF == 27:  
        break
        

      

video_capture.release()
cv2.destroyAllWindows()
