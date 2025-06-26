from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

CLASSES = ['forest', 'mountain', 'street']

def predict_image(img_path):
    model = load_model("demo_model.h5")
    img = image.load_img(img_path, target_size=(64, 64))
    img_array = np.expand_dims(img, axis=0) / 255.0
    pred = model.predict(img_array).argmax()
    return CLASSES[pred]
