from http import HTTPStatus
from lib.color import detect_and_classify_color
from lib.category import crop, classify
from lib.utils import url_to_cv2_image, base64_to_cv2_image


def get_colors(data):
    results = []

    if data["upload_type"] == 1:  # image_url
        opencv_image = url_to_cv2_image(data["image"])

    elif data["upload_type"] == 2:  # upload_image
        opencv_image = base64_to_cv2_image(data["image"])

    results = detect_and_classify_color(opencv_image)
    status_code = HTTPStatus.OK if results else HTTPStatus.NO_CONTENT

    return {"message": "success", "results": results}, status_code


def get_categories(data):
    results = []

    if data["upload_type"] == 1:  # image_url
        opencv_image = url_to_cv2_image(data["image"])

    elif data["upload_type"] == 2:  # upload_image
        opencv_image = base64_to_cv2_image(data["image"])

    cropped_results = crop(opencv_image)
    if not cropped_results:
        return {"message": "success", "results": []}, HTTPStatus.NO_CONTENT

    classes = classify([result["opencv_image"] for result in cropped_results])
    for index, cropped_result in enumerate(cropped_results):
        # output = classify(cropped_result["opencv_image"])
        # print(output)

        max_confidence = 0
        max_class = 0
        for category_class, confidence in enumerate(classes[index]):
            if confidence > max_confidence:
                max_confidence = confidence
                max_class = category_class

        results.append(
            {
                "bounding_box": cropped_result["bounding_box"],
                "confidence": float(max_confidence),
                "class": max_class,
            }
        )

    status_code = HTTPStatus.OK if results else HTTPStatus.NO_CONTENT

    return {"message": "success", "results": results}, status_code
