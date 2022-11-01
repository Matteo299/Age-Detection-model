import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# The face detector network is loaded using cv2.dnn.readNetFromCaffe and the model's layers and weights as passed its arguments.
modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "models/deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# Load the age detector model from disk
model = load_model('models/cnn_model.h5')

# Age ranges are defined
ranges = ['1-2','3-9','10-20','21-27','28-45','46-65','66-116']

# Capturing video or webcam
cap = cv2.VideoCapture('age_benchmark.mov')

# Final result is going to be written in the disk
result = cv2.VideoWriter('output/model.mov', 
                        cv2.VideoWriter_fourcc(*'MJPG'),
                        30, (1620,1080))

while True:
    
    # Read the frame
    _, img = cap.read()

    # Height and width of the image are extracted
    h, w = img.shape[:2]

    # To achieve the best accuracy I ran the model on BGR images resized to 300x300 
    # applying mean subtraction of values (104, 177, 123) for each blue, green and red channels correspondingly.
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,(300, 300), (104.0, 117.0, 123.0))
    net.setInput(blob)

    # Face detection is performed
    faces = net.forward()

    # For each face detected...
    for j in range(faces.shape[2]):
        confidence = faces[0, 0, j, 2]
        # If the confidence is above a certain threshold
        if confidence > 0.5:
            box = faces[0, 0, j, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")
            # Face is extracted
            img_face = img[y:y1, x:x1]
            # Tranformed to grayscale
            face = cv2.cvtColor(img_face,cv2.COLOR_BGR2GRAY)
            # Resized and reshaped to fit the input layer of the network
            face = cv2.resize(face,(200,200))
            face = face.reshape(1,200,200,1)
            # Normalized
            normalizer = ImageDataGenerator(rescale=1./255)
            facenorm = normalizer.flow(face)
            # Prediction is performed
            age = model.predict(facenorm)
            # A bounding box is drawn surrounding the face
            cv2.rectangle(img, (x, y), (x1, y1), (0,0,255), 2)
            # The estimated age range is printed
            cv2.putText(img,ranges[np.argmax(age)],(x1,y1),cv2.FONT_HERSHEY_DUPLEX,0.8,(255,255,255),2)
            
    # Display
    cv2.imshow('Age Detector', img)
    result.write(img)

    # Stop if escape key is pressed
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
        
# Release the VideoCapture object
cap.release()
result.release()
cv2.destroyAllWindows()