from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="./Models/6/model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

CLASS_NAMES = ["Apple Scab", "Black Rot", "Cedar Apple Rust", "Healthy"]

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.get("/ping")
async def ping():
    return "Hello, I am working!"

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    
    # Preprocess the image if needed
    # ...

    # Set the input tensor for the model
    interpreter.set_tensor(input_details[0]['index'], np.expand_dims(image, 0).astype(np.float32))

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    raw_prediction = output_data.tolist()
    probabilities = np.exp(raw_prediction) / np.sum(np.exp(raw_prediction))

    predicted_class_index = np.argmax(probabilities)
    predicted_class = CLASS_NAMES[predicted_class_index]
    
    return {"result": predicted_class}

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
