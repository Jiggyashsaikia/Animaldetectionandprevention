import cv2
import numpy as np
import serial

# Load the pre-trained MobileNet SSD model
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')

# Define the classes for MobileNet SSD
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# Define the classes of interest (animals and humans)
DETECTION_CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# Open a connection to the webcam
cap = cv2.VideoCapture(1)  # Use 0 for the default webcam, or change to the appropriate index

# Create a window for displaying the webcam feed
cv2.namedWindow('Real-Time Object Detection', cv2.WINDOW_NORMAL)

# Configure serial communication
ser = serial.Serial('COM3', 9600)  # Replace 'COM3' with the appropriate serial port on your system

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()

    # Check if the frame is empty
    if not ret or frame is None:
        continue

    # Perform object detection on the frame
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # Flag to check if a cow is detected
    cow_detected = False

    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        class_id = int(detections[0, 0, i, 1])

        if confidence > 0.2 and CLASSES[class_id] in DETECTION_CLASSES:
            label = f"{CLASSES[class_id]}: {confidence:.2f}"

            # Draw bounding box and label on the frame
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (startX, startY, endX, endY) = box.astype(int)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, label, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Check if the detected class is a cow
            if CLASSES[class_id] == "cow":
                cow_detected = True

    # Display the frame with object detections
    cv2.imshow('Real-Time Object Detection', frame)

    # Send a signal to ESP32 if a cow is detected
    if cow_detected:
        ser.write(b'CowDetected\n')

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('a'):
        break

# Release the webcam, close the window, and close the serial connection
cap.release()
cv2.destroyAllWindows()
ser.close()