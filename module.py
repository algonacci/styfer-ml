import tensorflow as tf
from dotenv import dotenv_values
from google.oauth2 import service_account
from google.cloud import storage
import matplotlib.pyplot as plt
import time

config = dotenv_values(".env")

CLOUD_STORAGE_BUCKET = config["CLOUD_STORAGE_BUCKET"]
credentials = service_account.Credentials.from_service_account_file(
    "styfer.json")

style_predict_model = "models/magenta_arbitrary-image-stylization-v1-256_int8_prediction_1.tflite"
style_transform_model = "models/magenta_arbitrary-image-stylization-v1-256_int8_transfer_1.tflite"

timestr = time.strftime("%Y%m%d-%H%M%S")


def upload_input_image_to_storage_bucket(image, filename):
    saved_image_path = "uploads/" + filename
    gcs = storage.Client(credentials=credentials)
    bucket = gcs.get_bucket(CLOUD_STORAGE_BUCKET)
    blob = bucket.blob(saved_image_path)
    blob.upload_from_filename(saved_image_path)
    blob.make_public()
    return blob.public_url


def upload_stylized_image_to_storage_bucket(image, filename):
    saved_image_path = "results/" + filename
    gcs = storage.Client(credentials=credentials)
    bucket = gcs.get_bucket(CLOUD_STORAGE_BUCKET)
    blob = bucket.blob(saved_image_path)
    blob.upload_from_filename(saved_image_path)
    blob.make_public()
    return blob.public_url


def load_image(path_to_img):
    img = tf.io.read_file(path_to_img)
    img = tf.io.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = img[tf.newaxis, :]
    return img


def preprocess_image(image, target_dim):
    shape = tf.cast(tf.shape(image)[1:-1], tf.float32)
    short_dim = min(shape)
    scale = target_dim / short_dim
    new_shape = tf.cast(shape * scale, tf.int32)
    image = tf.image.resize(image, new_shape)
    image = tf.image.resize_with_crop_or_pad(image, target_dim, target_dim)
    return image


def run_style_predict(preprocessed_style_image):
    interpreter = tf.lite.Interpreter(model_path=style_predict_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]["index"], preprocessed_style_image)
    interpreter.invoke()
    style_bottleneck = interpreter.tensor(
        interpreter.get_output_details()[0]["index"])()
    return style_bottleneck


def run_style_transform(style_bottleneck, preprocessed_content_image):
    interpreter = tf.lite.Interpreter(model_path=style_transform_model)
    input_details = interpreter.get_input_details()
    interpreter.allocate_tensors()
    interpreter.set_tensor(
        input_details[0]["index"], preprocessed_content_image)
    interpreter.set_tensor(input_details[1]["index"], style_bottleneck)
    interpreter.invoke()
    stylized_image = interpreter.tensor(
        interpreter.get_output_details()[0]["index"])()
    return stylized_image


def imshow(image, title=None):
    plt.figure(figsize=(3, 3))
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)
    plt.axis('off')
    plt.imshow(image)
    plt.savefig(fname="results/"+timestr+".jpg",
                bbox_inches='tight', pad_inches=0)
    imshow.filename = timestr + ".jpg"
    image_path = "results/" + timestr + ".jpg"
    return image_path
