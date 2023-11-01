import cv2
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

np.set_printoptions(suppress=True)

# Load the Keras model and class names
model = load_model("keras/keras_Model.h5", compile=False)
class_names = [line.strip() for line in open("keras/labels.txt", "r")]

# Load the image using OpenCV
image = cv2.imread("images/img1.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB format

# Define the size and resize the image
size = (224, 224)
image = Image.fromarray(image)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

image_array = np.asarray(image)

# Normalize the image data
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Prepare the data for model prediction
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
data[0] = normalized_image_array

# Make predictions using the model
prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]

# Get the dimensions of the bounding box (for example, you may obtain these from your object detection model)
x, y, w, h = 100, 100, 50, 50  # Replace these with the actual bounding box coordinates

# Draw the bounding box on the image
cv2.rectangle(image_array, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green rectangle

# Display the class name and confidence score on the image
cv2.putText(image_array, f"Class: {class_name[2:]}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
cv2.putText(image_array, f"Confidence Score: {confidence_score:.2f}", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

print("Class:", class_name[2:])
print("Confidence Score:", confidence_score)

# Create a Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Detect faces in the image
faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Draw rectangles around the detected faces
for (fx, fy, fw, fh) in faces:
    cv2.rectangle(image_array, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)  # Blue rectangle for faces

# Display the image with the bounding box and information
cv2.imshow("Image with Bounding Box and Face Detection", cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()
