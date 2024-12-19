import face_recognition
from PIL import Image
import numpy as np

# Load the image
image_path = "faces/jyoti.jpg"
jyoti_image = face_recognition.load_image_file(image_path)

# Convert image to RGB
jyoti_image = Image.fromarray(jyoti_image).convert("RGB")
jyoti_image = np.array(jyoti_image)

# Get face encodings
jyoti_encoding = face_recognition.face_encodings(jyoti_image)[0]

print("Face encoding successfully generated!")
