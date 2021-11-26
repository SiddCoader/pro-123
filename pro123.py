import cv2
import numpy as np
from numpy.lib.arraysetops import intersect1d
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os, ssl, time

if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
    getattr(ssl, '_create_unverified_context', None)): 
    ssl._create_default_https_context = ssl._create_unverified_context

#Uploading image data
from google.colab import files
data_to_load = files.upload()

#Uploading labels data
from google.colab import files
label_to_load = files.upload()

X = np.load('image.npz')['arr_0']
y = pd.read_csv("labels.csv")["labels"]
print(pd.Series(y).value_counts())
classes = ['A', 'B', 'C', 'D', 'E','F', 'G', 'H', 'I', 'J', "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
nclasses = len(classes)

#roi = Region of Interest
roi = gray[upper_left[1]:bottom_right[1],
upper_left[0]:bottom_right[0]]

image_bw = im_pil.convert('L')
image_bw_resized = image_bw.resize((28,28), Image.ANTIALIAS)

image_bw_resized_inverted = PIL.ImaheOps.invert(image_bw_resized)
pixel_filter = 20

min_pixel = np.percentile(image_bw_resized_inverted, pixel_filter)

image_bw_resized_inverted_scaled = np.clip(image_bw_resized_inverted-min_pixel, 0, 255)
max_pixel = np.max(image_bw_resized_inverted)

image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel

test_sample = np.asarray(image_bw_resized_inverted_scaled).reshape(1,784)
test_pred = clf.predict(test_samples)
print("Predicated class is: ",test_pred)