# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import time
import json
import requests
import numpy as np

from PIL import Image

from crnn.keys import alphabetChinese
from crnn.util import resizeNormalize, strLabelConverter

# ---------------Params---------------

IMAGE_PATH = "./demo_images/test.jpg"
CRNN_API_URL = "http://text_recognition:8501/v1/models/crnn:predict"

# ---------------Alphabet---------------

alphabet = alphabetChinese
nclass = len(alphabet) + 1

# ---------------Process image---------------

image = cv2.imread(IMAGE_PATH)
image = Image.fromarray(image)
image = image.convert('L')
image = resizeNormalize(image, 32)
image = image.astype(np.float32)
image = np.array([image])

# ---------------Build post---------------

post_json = {
    "instances": [{
        "input_image": image.tolist()
    }]
}

# ---------------Test---------------

t0 = time.time()
response = requests.post(CRNN_API_URL, data=json.dumps(post_json))
print("forward time : {}".format(time.time() - t0))
response.raise_for_status()
prediction = response.json()["predictions"]
print(prediction)
raw = strLabelConverter(prediction[0], alphabet)
print(raw)
