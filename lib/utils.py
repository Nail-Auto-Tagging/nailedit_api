import base64
import cv2
import numpy as np
import urllib.request

    
def url_to_cv2_image(url: str):
	resp = urllib.request.urlopen(url)
	image = np.asarray(bytearray(resp.read()), dtype="uint8")
	cv2_image = cv2.imdecode(image, cv2.IMREAD_COLOR)
	return cv2_image

def base64_to_cv2_image(base64_string):
    arr = base64_string.split(";base64,")
    encoded_img = np.fromstring(base64.b64decode(arr[1]), dtype = np.uint8)
    cv2_image = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
    return cv2_image