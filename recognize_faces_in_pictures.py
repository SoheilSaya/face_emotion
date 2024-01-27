import face_recognition
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json

# Load the emotion recognition model
json_file = open('fer.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("fer.h5")
print("Loaded emotion recognition model from disk")

# Define emotion labels
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load the jpg files into numpy arrays
biden_image = face_recognition.load_image_file("biden.jpg")
obama_image = face_recognition.load_image_file("obama.jpg")
iman_image = face_recognition.load_image_file("iman.jpg")
chuck_image = face_recognition.load_image_file("chuck.jpg")
kim_image = face_recognition.load_image_file("kim.jpg")


# Get the face encodings for each face in each image file
biden_face_encoding = face_recognition.face_encodings(biden_image)
obama_face_encoding = face_recognition.face_encodings(obama_image)
iman_face_encoding = face_recognition.face_encodings(iman_image)
chuck_face_encoding = face_recognition.face_encodings(chuck_image)
kim_face_encoding = face_recognition.face_encodings(kim_image)



known_faces = [
    biden_face_encoding[0],
    obama_face_encoding[0],
    iman_face_encoding[0],
    chuck_face_encoding[0],
    kim_face_encoding[0]
]



# Create a dictionary mapping names to their corresponding face encodings
faces_index = {
    0:"Biden",
    1:"Obama",
    2:"Iman",
    3:'chuck',
    4:'kim'
}



def face_emotion(picture_name):


    # Get the bounding boxes of detected faces
    unknown_image = face_recognition.load_image_file(picture_name)
    unknown_face_locations = face_recognition.face_locations(unknown_image)


    



        
    print('_____________________________________________')
    print(unknown_face_locations)
    # If unknown face is detected, perform emotion recognition
    if unknown_face_locations:
        for i,(top, right, bottom, left) in enumerate(unknown_face_locations) :
            # Crop the face from the image
            cropped_face = unknown_image[top:bottom, left:right]

            
            
            #unknown_image = face_recognition.load_image_file(picture_name)
            unknown_face_encoding = face_recognition.face_encodings(unknown_image)
            # Compare the unknown face encoding with the known faces
            results = face_recognition.compare_faces(known_faces, unknown_face_encoding[0])
            if not biden_face_encoding or not obama_face_encoding or not iman_face_encoding:
                print("No faces were found in one or more images. Aborting...")
                quit()
    
            
            person_names=[]
            match_found=False
            
            for index, face_names in faces_index.items():
                if results[index]:
                    person_name=faces_index[index]
                    person_names.append(person_name)
                    print('this is',person_name)
                    match_found=True
            if not match_found:
                print('NO FRENLY FACES')

            
            
            
            # Resize the cropped face to match the input size of the emotion recognition model
            cropped_face_resized = cv2.resize(cropped_face, (48, 48))
            # Convert the cropped face to grayscale
            cropped_face_gray = cv2.cvtColor(cropped_face_resized, cv2.COLOR_BGR2GRAY)
            # Normalize the cropped face
            cropped_face_normalized = cropped_face_gray / 255.0
            # Reshape the cropped face to match the input shape of the emotion recognition model
            cropped_face_reshaped = cropped_face_normalized.reshape(1, 48, 48, 1)
            # Predict the emotion using the emotion recognition model
            emotion_prediction = loaded_model.predict(cropped_face_reshaped)
            # Get the index of the predicted emotion
            emotion_index = np.argmax(emotion_prediction)
            # Get the label of the predicted emotion
            predicted_emotion = labels[emotion_index]
            # Draw bounding box and label for the detected face with predicted emotion
            cv2.rectangle(unknown_image, (left, top), (right, bottom), (0, 255, 0), 1)
            cv2.putText(unknown_image, predicted_emotion, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
            print(person_name,'is', predicted_emotion)
            #print('_____________________________________________')

    # Display the image with bounding boxes and emotion labels
    # Display the cropped face image
    # unknown_image_resized = cv2.resize(unknown_image, (400, 400))
    # cropped_face_resized = cv2.resize(cropped_face, (400, 400))

    # # Concatenate images horizontally
    # combined_image = np.concatenate((cropped_face_resized, unknown_image_resized), axis=1)

    # # Display the combined image
    # cv2.imshow('Emotion Recognition', combined_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
import time

t0=time.time()
face_emotion('obama2.jpg')
print('time took for obama2',time.time()-t0)
t0=time.time()
face_emotion('iman2.jpg')
print('time took for iman',time.time()-t0)
t0=time.time()
face_emotion('chuck2.jpg')
print('time took for chuck2',time.time()-t0)
t0=time.time()
face_emotion('kim2.jpg')
print('time took for kim2',time.time()-t0)
t0=time.time()
face_emotion('biden2.jpg')
print('time took for biden2',time.time()-t0)
t0=time.time()
face_emotion('two_people.jpg')
print('time took for two_people',time.time()-t0)