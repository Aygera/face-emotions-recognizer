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

# Run on CPU if needed
tensorflow.device('/cpu:0')


def draw_image(image):
    cv2.imshow("output", image)  # Show image
    cv2.waitKey(0)


# function accepts image_urls and returns list of detected emotions (if face itself was detected)
def detect_emotions(image_url):
    # parameters for loading data and images
    emotion_model_path = './models/emotion_model.hdf5'
    emotion_labels = get_labels('fer2013')

    # loading model for face detection
    detector = MTCNN()
    # loading model for emotion classification
    emotion_classifier = load_model(emotion_model_path)

    # getting input model shapes for inference
    emotion_target_size = emotion_classifier.input_shape[1:3]

    # tries to open image by URL
    try:
        img = Image.open(requests.get(image_url, stream=True).raw)
    except:
        return "Couldn't find image with the given url"

    # Converts Image PIL object into opencv format
    open_cv_image = np.array(img)
    bgr_image = open_cv_image[:, :, ::-1].copy()
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    # MTCCN module which detects faces on image (pretrained model was used)
    faces = detector.detect_faces(rgb_image)

    # stores final result by all faces found in an image
    final_result = []
    # id of faces
    count = 0

    # iterates over detected faces
    for face_coordinates in faces:

        # bounding boxes of the face
        bbox = face_coordinates['box']
        x1, x2, y1, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

        # crop image by bbox of detected face
        cropped_image = bgr_image[x2:x2 + y2, x1:x1 + y1]
        # draw_image(bgr_image, 0, 0, 0, 0)

        # converts to grayscale
        gray_face = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        try:
            # resize image in a size that model accepts
            gray_face = cv2.resize(gray_face, emotion_target_size)
        except:
            continue

        # adds face_id
        intermediate_results = [['face_id: ', count]]

        # converts pixels into floating point numbers
        gray_face = preprocess_input(gray_face, True)

        # reshapes cropped gray image to feed emotion classifier
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)

        # the face array is run through classifier and outputs predictions for all 7 emotions
        emotion_prediction = emotion_classifier.predict(gray_face)

        # sorts list by id
        results = [[i, r] for i, r in enumerate(emotion_prediction[0])]
        # rearranges results in descending order by emotions probs
        results.sort(key=lambda x: x[1], reverse=True)
        for r in results:
            intermediate_results.append([emotion_labels[r[0]], str(r[1])])

        # adds face results to list
        intermediate_results.append(['bbox starting x1,y1 and x2, y2: ', x2, x2 + y2, x1, x1 + y1])
        count += 1
        final_result.append([intermediate_results])

    return final_result

# URLs of images for testing purpose

# url_img = "https://cdn.trendhunterstatic.com/thumbs/human-emotions-photo-series.jpeg"
# url_img = "/home/aigerim/Downloads/human-emotions-photo-series.jpeg"
# url_img = "/home/aigerim/Downloads/pexels-tomaz-barcellos-1987301.jpg"
# result = detect_emotions(url_img)

# print(result)
