import cv2
from mtcnn import MTCNN
import numpy as np
import urllib.request
import tensorflow
from tensorflow.keras.models import load_model
from utils.datasets import get_labels
from utils.inference import apply_offsets
from utils.preprocessor import preprocess_input
from PIL import Image
import requests

tensorflow.device('/cpu:0')


def detect_emotions(image_url):
    # parameters for loading data and images
    emotion_model_path = './models/emotion_model.hdf5'
    emotion_labels = get_labels('fer2013')

    # hyper-parameters for bounding boxes shape
    frame_window = 10
    emotion_offsets = (20, 40)

    # loading models
    detector = MTCNN()
    emotion_classifier = load_model(emotion_model_path)

    # getting input model shapes for inference
    emotion_target_size = emotion_classifier.input_shape[1:3]

    # starting lists for calculating modes
    emotion_window = []
    try:
        img = Image.open(requests.get(image_url, stream=True).raw)
    except:
        return "Couldn't find image with the given url"

    bgr_image = np.array(img)

    faces = detector.detect_faces(bgr_image)

    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

    final_result = []
    count = 0

    for face_coordinates in faces:

        bbox = face_coordinates['box']
        x1, x2, y1, y2 = apply_offsets((bbox[0], bbox[1], bbox[2], bbox[3]), emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue
        intermidiate_resutls = [['face_id: ',count]]
        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]
        emotion_window.append(emotion_text)

        results = [[i, r] for i, r in enumerate(emotion_prediction[0])]
        results.sort(key=lambda x: x[1], reverse=True)
        for r in results:
            intermidiate_resutls.append([emotion_labels[r[0]], str(r[1])])

        intermidiate_resutls.append(['bbox: ', x1, y1, x2, y2])
        count +=1
        final_result.append([intermidiate_resutls])

    return final_result
'''
url_img = "https://cdn.trendhunterstatic.com/thumbs/human-emotions-photo-series.jpeg"
#url_img = "https://images.pexels.com/photos/1987301/pexels-photo-1987301.jpeg?cs=srgb&dl=pexels-tomaz-barcellos-1987301.jpg&fm=jpg"
result = detect_emotions(url_img)

print(result)

'''