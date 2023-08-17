from sklearn.datasets import fetch_openml
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import cv2
from PIL import Image
import PIL.ImageOps

X = np.load('image.npz')['arr_0']
y = pd.read_csv("labels.csv")["labels"]
print(pd.Series(y).value_counts())
classes = ['A', 'B', 'C', 'D', 'E','F', 'G', 'H', 'I', 'J', "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
nclasses = len(classes)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 3500, test_size = 500)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

model = LogisticRegression(multi_class='multinomial', solver='saga')
model.fit(X_train, y_train)
predictions = model.predict(X_test)

score = accuracy_score(y_test, predictions)
print(score)

cap=cv2.VideoCapture(0)
while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Drwaing a box in the center of the video
    height, width = gray.shape
    upper_left = (int(width/2)-56, int(height/2)-56)
    bottom_right = (int(width/2)+56, int(height/2)+56)
    cv2.rectangle(gray, upper_left, bottom_right, (0, 0, 0),2)
    #To consider the area inside the box for detecting the digit
    #ROI = Region of Interest
    roi = gray[upper_left[1]:bottom_right[1],upper_left[0]:bottom_right[0]]
    #Converting cv2 image to PIL format
    im_pil = Image.fromarray(roi)
    im_bw = im_pil.convert('L')
    im_bw_resized = im_bw.resize((22,30), Image.ANTIALIAS)
    #Invert the image
    im_bw_resized_inverted = PIL.ImageOps.invert(im_bw_resized)
    pixel_filter = 20
    min_pixel = np.percentile(im_bw_resized_inverted, pixel_filter)
    im_bw_resized_inverted_scaled = np.clip(im_bw_resized_inverted-min_pixel, 0, 225)
    max_pixel - np.max(im_bw_resized_inverted)
    im_bw_resized_inverted_scaled = np.asarray(im_bw_resized_inverted_scaled)/max_pixel
    test_sample = np.array(im_bw_resized_inverted_scaled).reshape(1,660)
    test_pred = model.preict(test_sample)
    print('Predicted number is: ', test_pred)
    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
