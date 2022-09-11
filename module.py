import tensorflow as tf
from dotenv import dotenv_values
from google.oauth2 import service_account
from google.cloud import storage

config = dotenv_values(".env")

CLOUD_STORAGE_BUCKET = config["CLOUD_STORAGE_BUCKET"]
credentials = service_account.Credentials.from_service_account_file(
    "styfer.json")

style_predict_model = "models/magenta_arbitrary-image-stylization-v1-256_int8_prediction_1.tflite"
style_transform_model = "models/magenta_arbitrary-image-stylization-v1-256_int8_transfer_1.tflite"


def upload_input_image_to_storage_bucket(image, filename):
    saved_image_path = "uploads/" + filename
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
