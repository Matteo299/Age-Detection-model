# FaceDetectionComparator
Age detection is the process of automatically discerning the age of a person solely from a photo of their face.<br> In this project, my goal is to build an age classification algorithm using images of people’s faces (taken from public datasets found online).<br>
Faces are detected using the face detection model of the DNN module of OpenCV. In the first part of the project, we demonstrated that it works better than Haar cascade and is significantly less resource consuming than MTCNN.<br>
Once obtained the bounding box coordinates of the faces, they are extracted ignoring the rest of the image/frame. Doing so allows the age detector to focus solely on the person’s face and not any other irrelevant “noise” in the image. <br>The face ROI is then passed through the model, yielding the actual age prediction.<br>
Look at the report folder for more info on how to run the code and which dataset to download.
