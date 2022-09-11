import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import module as md
import time

app = Flask(__name__)
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['RESULT_FOLDER'] = 'results/'

timestr = time.strftime("%Y%m%d-%H%M%S")


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


@app.route("/")
def index():
    return {
        "status_code": 200,
        "message": "Success!"
    }


@app.route("/transfer", methods=["POST"])
def transfer():
    if request.method == "POST":
        input_image = request.files["input_image"]
        style_number = request.form["style_number"]
        if input_image and allowed_file(input_image.filename):
            transfer.filename = secure_filename(
                timestr + "_" + input_image.filename)
            print(transfer.filename)
            input_image.save(os.path.join(
                app.config['UPLOAD_FOLDER'], transfer.filename))
            image = os.path.join(
                app.config['UPLOAD_FOLDER'], transfer.filename)
            uploaded_image_url = md.upload_input_image_to_storage_bucket(
                image=input_image, filename=transfer.filename)
            content_image = md.load_image(image)
            style_image = md.load_image("data/{}.jpg".format(style_number))
            preprocessed_content_image = md.preprocess_image(
                content_image, 384)
            preprocessed_style_image = md.preprocess_image(
                style_image, 256)
            style_bottleneck = md.run_style_predict(preprocessed_style_image)
            stylized_image = md.run_style_transform(
                style_bottleneck, preprocessed_content_image)
            stylized_image_path = md.imshow(
                stylized_image, filename=transfer.filename)
            stylized_image_url = md.upload_stylized_image_to_storage_bucket(
                image=stylized_image_path, filename=transfer.filename)
            json = {
                "status_code": 200,
                "message": "Success uploading image!",
                "input_image": uploaded_image_url,
                "stylized_image": stylized_image_url
            }
            return jsonify(json)
        else:
            return {
                "status_code": 400,
                "message": "Please upload an image!"
            }
    else:
        return {
            "status_code": 400,
            "message": "Use POST method!"
        }


@app.errorhandler(400)
def bad_request(error):
    return {
        "status_code": 400,
        "error": error,
        "message": "Client side error!"
    }, 400


@app.errorhandler(404)
def not_found(error):
    return {
        "status_code": 404,
        "message": "URL not found!"
    }, 404


@app.errorhandler(405)
def method_not_allowed(error):
    return {
        "status_code": 405,
        "message": "Check the exisiting allowed method!"
    }, 405


@app.errorhandler(500)
def internal_server_error(error):
    return {
        "status_code": 500,
        "message": "Server error"
    }, 500


if __name__ == "__main__":
    app.run(debug=True,
            host="0.0.0.0",
            port=int(os.environ.get("PORT", 8080)))
