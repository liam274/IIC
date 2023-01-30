from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import requests
import os
from io import BytesIO
from flask import Flask, request
from flask_cors import CORS

app=Flask(__name__)
CORS(app)

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()
def LoadImage(URL,height=224,width=224):
    return ImageOps.fit(Image.open(BytesIO(requests.get(URL).content)).convert("RGB"),(width,height),Image.Resampling.LANCZOS)
@app.route("/prediction/", methods=['GET', 'POST'])
def keras():
    #Get all the values in your POST request. 
    image = request.args.get('url')
    threshold = int(request.args.get("threshold"))
    #Follow all the neccessary steps to get the prediction of your image. 
    image = loadImage(image)
    #turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    # if prediction.argmax() >= threshold/100:
    #     Id, recognized_object = labels[prediction.argmax()] # the label that corresponds to highest prediction
    # else:
    #     recognized_object = "unknown"

    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    print(index, class_name, confidence_score, prediction)
    if confidence_score >= threshold/100:
        return class_name, 200
    else:
        return "not known", 200
    #Return the prediction and a 200 status
    #return recognized_object, 200

@app.route('/')
def hello_world():
    return 'Hello, World!', 200
if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
"""
# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
image = Image.open("<IMAGE_PATH>").convert("RGB")

# resizing the image to be at least 224x224 and then cropping from the center
size = (224, 224)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# turn the image into a numpy array
image_array = np.asarray(image)

# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Load the image into the array
data[0] = normalized_image_array

# Predicts the model
prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]

# Print prediction and confidence score
print("Class:", class_name[2:], end="")
print("Confidence Score:", confidence_score)"""#Maybe no use
