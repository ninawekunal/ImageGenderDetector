import pickle
from PIL import Image
import numpy as np
import pandas as pd
import cv2
import matplotlib
import matplotlib.pyplot as plt

# Loading models
haar = cv2.CascadeClassifier('models/haarcascade_classifier.xml')
mean = pickle.load(open('models/mean_pre_processing.pickle', 'rb'))
model_svm = pickle.load(open('models/model_svm_ideal.pickle', 'rb'))
model_pca = pickle.load(open('models/pca_50.pickle', 'rb'))

print("Models loaded successfully..")

# Default settings
gender_predict = ['Male', 'Female']
font = cv2.FONT_HERSHEY_SIMPLEX


def pipeline_model(path, filename, color='bgr'):
    # Step 1 - Read image in cv2: bgr format
    img = cv2.imread(path)

    # Step 2 - Convert to grayscale
    color = 'bgr'
    if color == 'bgr':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Step 3 - Crop the face using Haar Cascade Classifier

    faces = haar.detectMultiScale(gray, 1.5, 3)
    for x, y, w, h in faces:
        # Draw Rectangle.
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)
        roi = gray[y:y+h, x:x+w]    # Crop the image.

    # Step 4 - Normalizing the data.
        roi = roi/255.0

    # Resize image
        if roi.shape[1] >= 100:
            roi_resize = cv2.resize(roi, (100, 100), cv2.INTER_AREA)  # SHRINK
        else:
            roi_resize = cv2.resize(
                roi, (100, 100), cv2.INTER_CUBIC)   # ENLARGE

    # Step 5 - Flatten the data.
        roi_reshape = roi_resize.reshape(1, -1)

    # Step 6 - Subtract from mean.
        roi_mean = roi_reshape - mean

    # Step 7 - Get Eigen Image using PCA.
        eigen_image = model_pca.transform(roi_mean)

    # Step 8 - Pass to ML SVM Model.
        results = model_svm.predict_proba(eigen_image)[0]

    # Step 9 - Predict
        # Pick the gender whose probability is highest. returns either 0 or 1
        predict = results.argmax()
        score = results[predict]

    # Step 10 - Generate result output
        text = "%s : %0.2f" % (gender_predict[predict], score)
        cv2.putText(img, text, (x, y), font, 1, (255, 255, 0), 2)
    cv2.imwrite('./static/predict/{}'.format(filename), img)
