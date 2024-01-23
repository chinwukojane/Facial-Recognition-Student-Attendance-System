# Import libraries
import cv2
import os
import uuid
from datetime import datetime, timedelta
from pathlib import Path
import sys
from keras.models import load_model
from PIL import Image, ImageOps #Install pillow instead of PIL
import numpy as np

# get system data and time
time_now = datetime.now()
# get system time
current_time = time_now.strftime("%H:%M:%S")
# get system date
current_date = time_now.strftime("%d-%m-%Y")

# variables store image count values
no_face_count: int = 0

record_attendance = 0
prev_detect = ''
last_played = ''
condition = ''
monitor_attendance = 0

# Create Folder Structures
POS_PATH_TRAIN = os.path.join('student_data', 'take')

POS_PATH_TRAIN2 = os.path.join('student_data', 'student_attendance')
if not os.path.exists(POS_PATH_TRAIN2):
    os.makedirs(POS_PATH_TRAIN2)

# Load the model
model = load_model('CNN_model.h5', compile=False)

# Load the labels
class_names = open('labels.txt', 'r').readlines()

# check if test and train image directories exists and create if null
if not os.path.exists(POS_PATH_TRAIN):
    os.makedirs(POS_PATH_TRAIN)

 # read previous attendance list
my_file = Path(POS_PATH_TRAIN2 + '/' + current_date + '_Attendance_Data.txt')
# clear previous read data
last_student_id = ''
# check if file exists
if my_file.exists():
    # read labels text file into list
    with open(POS_PATH_TRAIN2 + '/' + current_date + '_Attendance_Data.txt') as read_labels:
        # read all data
        last_student_id = read_labels.read()

def register_attendance(condition):
    global prev_detect, record_attendance, monitor_attendance

    # check if string exists
    if prev_detect.find(condition) != -1 and last_student_id.find(condition) == -1:
        # countinue verfication counting if less than 10
        if record_attendance <= 10:
            # continue count if recognition is the same
            record_attendance = record_attendance + 1
            if record_attendance > 10:
                # store attendance already take value
                record_attendance = 11
            # if verification count = 10
            if record_attendance == 10:
                # read previous attendance list
                my_file = Path(POS_PATH_TRAIN2 + '/' + current_date + '_Attendance_Data.txt')
                # clear previous read data
                student_id = ''
                monitor_attendance = 0
                # check if file exists
                if my_file.exists():
                    # read labels text file into list
                    with open(POS_PATH_TRAIN2 + '/' + current_date + '_Attendance_Data.txt') as read_labels:
                        # read all data
                        student_id = read_labels.read()
                # take new attendance if student has not taken yet
                if student_id.find(condition) == -1 or student_id == '':
                    new_info = str('\nStudent ID: ' + condition + ' Time: ' + current_time + ' Date: ' + current_date)
                    # Gets the new value in the tkinter text box
                    with open(POS_PATH_TRAIN2 + '/' + current_date + '_Attendance_Data.txt', "a") as text_file:
                        text_file.write(new_info)
                    print('Attendance taken for ', condition)
                    monitor_attendance = 1

                # check if student has taken attendance already
                if student_id.find(condition) != -1:
                    # store attendance already take value
                    record_attendance = 11

                # save last face condition
                last_played = condition

    # check if string does not exist
    if prev_detect.find(condition) == -1:
        record_attendance = 0
        if last_student_id.find(condition) != -1:
            student_id = condition
            monitor_attendance = 3
        else:
            monitor_attendance = 0


    # save last recognition
    prev_detect = condition

    return

def predict_image():
    global condition, monitor_attendance
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)


    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = Image.open('C:/Users/Omen/PycharmProjects/test/venv/PROJECT/student_data/take/a.jpg').convert('RGB')

    # resize the image to a 224x224 with the same strategy as in TM2:
    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    print('Student Number:', class_name, end='')
    print('Confidence score:', confidence_score)

    condition = str(class_name)
    condition = condition.replace('\n', '')
    register_attendance(condition)

    return


import tensorflow as tf
# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Establish a connection to the webcam
cap = cv2.VideoCapture(0)
# Create 1 minutes interval timer
period = timedelta(minutes=1)
next_time = datetime.now() + period
# read camera video feed
while cap.isOpened():
    # read webcam
    ret, frame = cap.read()
    # Cut down frame to 250x250px
    frame = frame[120:120 + 250, 200:200 + 250, :]
    # resize video frame
    frame = cv2.resize(frame, (500, 500))
    # Convert to grayscale
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(frame, 1.1, 4)
    # Get the number of detected face values
    num_faces: int = int(len(faces))
    print('Number of faces: ', num_faces)

    img = os.path.join(POS_PATH_TRAIN, 'a.jpg')
    image = cv2.imread(img)
    image = np.resize(image, (224, 224, 3))
    image = np.array(image).astype('float32')
    image = image.reshape(1, 224, 224, 3)
    image = image / 255
    print(np.shape(image))
    tf.convert_to_tensor(image)
    pred = model.predict(image)
    print(pred)
    pred_index = np.argmax(pred)
    print("Prediction2: ", class_names[pred_index])

    # Monitor inactive face capture session
    if num_faces >= 1:
        no_face_count = 0
    if num_faces <= 0:
        if next_time <= datetime.now():
            next_time += period
            no_face_count = no_face_count + 1

    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        # draw the rectangle on the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # gray value
        gray_frame = frame[y:y + h, x:x + w]
        # crop the face image to 48 x 48 pixel
        cropped_img = cv2.resize(gray_frame, (224, 224))

        # Collect positives
        if num_faces == 1:
            if monitor_attendance == 0:
                cv2.putText(frame, 'Please wait! capturing',
                            (x - 60, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            if monitor_attendance == 1:
                cv2.putText(frame, 'Attendance Captured!',
                            (x - 60, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            if monitor_attendance == 3:
                cv2.putText(frame, 'Attendance Already Captured!',
                            (x - 60, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            # Create the unique file path
            imgname_train = os.path.join(POS_PATH_TRAIN, 'a.jpg')
            # Write out train image
            cv2.imwrite(imgname_train, cropped_img)
            if num_faces >= 1:
                predict_image()
            # reset number of no face count
            no_face_count = 0

    # Show image back to screen
    cv2.imshow('Student Attendance', frame)
    # break using the q key on the keyboard
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # break if no face is detected after 1 minute
    if cv2.waitKey(1) & no_face_count == 60:
        break

# Release the webcam
cap.release()
# Close the image show frame
cv2.destroyAllWindows()
