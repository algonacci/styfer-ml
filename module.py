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
