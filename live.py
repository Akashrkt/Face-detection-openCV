import cv2

cap = cv2.VideoCapture(0) # 0 specify camera no (default)

# Create the haar cascade
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while(True): # capture frame from video and process until break
	# Capture frame-by-frame
	ret, frame = cap.read() # frame holds image data ret indicates frame was sucessesfully read

	# Our operations on the frame come here
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #conversion to gray scale

	# Detect faces in the image
	faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(30, 30)
		#flags = cv2.CV_HAAR_SCALE_IMAGE
	)

	print("Found {0} faces!".format(len(faces))) #print no of face found in current frame

	# Draw a rectangle around the faces
	for (x, y, w, h) in faces:
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) # 2 is thickness


	# Display the resulting frame
	cv2.imshow('frame', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'): # q to stop capturing frame
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


