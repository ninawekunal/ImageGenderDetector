# ImageGenderDetector

**Objective:** To Develop a Face recognition project that is driven by data and machine learning models which can detect the face and classify the gender of the person in the uploaded image with an accuracy of 80% and can also work on multiple faces and video as well.

**Idea:** 
  - As soon as the user converts the image, the image is converted into grayscale, 
  - the face is then cropped using a classifier and 
  - then it is finally passed into a machine learning model to predict the gender. 
  - All of this will be dumped into the flask app later on. 

**Dataset:** Images dataset of celebrities that I have on my system. 

**Libraries:**
 - Numpy
 - Pandas
 - OpenCV
 - MatplotLib
 - Sklearn

**Modules:**
  - Front-end is developed in 
      - Html5 + CSS3
      - Bootstrap
      - Flask
  - Backend is in python

**Tools:**
- Machine Learning techniques like:
  - Support Vector Machine (SVM) for model training,
  - Principal Component Analysis(PCA) for Eigen Images,
  - Haar Cascade Classifier to detect a face,
  - Grid Search to get the best parameters.
  - Model evaluation techniques like:
    - Confusion Matrix,
    - Classification Report,
    - Kappa Score,
    - ROC and AUC (Probability).
- Flask modules adding soon.


Everything is running and dumped into the flask web server.

**Limitations:** 
  - The dataset contains the images of mostly hollywood celebrities and might not work well with people of different ethnicity.
  - I did not do any check to see if the uploaded image is of a person or not.
  - Sometimes in a video, it detects objects as faces.
