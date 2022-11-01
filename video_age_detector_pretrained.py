import cv2
import numpy as np

# The face detector network is loaded using cv2.dnn.readNetFromCaffe and the model's layers and weights as passed its arguments.
modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "models/deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# Represent the 8 age classes of this CNN probability layer
AGE_INTERVALS = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)',
                 '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']

# The age detector network is loaded using cv2.dnn.readNetFromCaffe and the model's layers and weights as passed its arguments.
AGE_MODEL = 'models/deploy_age.prototxt'
AGE_PROTO = 'models/age_net.caffemodel'
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_net = cv2.dnn.readNetFromCaffe(AGE_MODEL, AGE_PROTO)

# Final result is going to be written in the disk
result = cv2.VideoWriter('output/model_pretrained.mov', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         30, (1620,1080))

# Capturing video or webcam
cap = cv2.VideoCapture("age_benchmark.mov")

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
            blob2 = cv2.dnn.blobFromImage(
            image=img_face, scalefactor=1.0, size=(227, 227), 
            mean=MODEL_MEAN_VALUES, swapRB=False
            )
            # Forward propagation is performed
            age_net.setInput(blob2)
            age_preds = age_net.forward()
            i = age_preds[0].argmax()
            age = AGE_INTERVALS[i]
            # A bounding box is drawn surrounding the face
            cv2.rectangle(img, (x, y), (x1, y1), (0,0,255), 2)
            # The estimated age range is printed
            cv2.putText(img,str(age),(x1,y1),cv2.FONT_HERSHEY_DUPLEX,0.8,(255,255,255),2)
            
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