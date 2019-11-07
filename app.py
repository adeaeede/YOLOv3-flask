from flask import Flask
from flask import jsonify, request, make_response, send_file
from keras.models import model_from_json
from yolov3 import inference
import cv2
import io
import base64
import numpy as np


def init_model():
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("model.h5")
    print("Loaded model from disk")
    return model


app = Flask(__name__)
_model = init_model()


@app.route('/api/v1/object_detection/', methods=['POST'])
def inference_single():
    if request.content_length > 4194304:
        return make_response(('Exceeded maximal content size', 413))
    imageString = base64.b64decode(request.json['image'])
    nparr = np.fromstring(imageString, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    probabilities, image = inference.inference_single(_model, image)
    _, img_encoded = cv2.imencode('.jpg', image)
    # Something does not work quite with broadcasting see line 197 in bound_box.py
    return send_file(io.BytesIO(img_encoded), as_attachment=True, attachment_filename='image_detected.jpg',
                     mimetype='image/jpg')


if __name__ == '__main__':
    app.run(debug=False, threaded=False)
