import cv2
from roboflow import Roboflow

# Initialize Roboflow
rf = Roboflow(api_key="SQ7otNG7MCDy8UPFj81t")
project = rf.workspace().project("pbl-final")
model = project.version(2).model

# Load an image using OpenCV
image = cv2.imread("images/img1.jpg")

# Predict on the image
result = model.predict("images/img1.jpg", confidence=40, overlap=30).json()

# Extract the predictions
predictions = result['predictions']

model.predict("images/img1.jpg", confidence=40, overlap=30).save("prediction.jpg")

# Iterate through the predictions and draw bounding boxes
for prediction in predictions:
    class_name = prediction['class']
    confidence = prediction['confidence']
    bbox = {
        'x': prediction['x'],
        'y': prediction['y'],
        'width': prediction['width'],
        'height': prediction['height']
    }

    print(class_name)
    print(confidence)

    x_min, y_min, x_max, y_max = int(bbox['x']), int(bbox['y']), int(bbox['x'] + bbox['width']), int(bbox['y'] + bbox['height'])

    # Draw bounding box
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Display class name and confidence
    label = f"{class_name}: {confidence:.2f}"
    cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Create a Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Convert the image to grayscale for face detection
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the grayscale image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Draw rectangles around the detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Increase the window size
cv2.namedWindow("Annotated Image with Face Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Annotated Image with Face Detection", 800, 600)

# Display the annotated image with faces in a window
cv2.imshow("Annotated Image with Face Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()