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
    try:
        imageString = base64.b64decode(request.json['image'])
        thresh = request.json['obj_thresh']
        nparr = np.fromstring(imageString, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        probabilities, image = inference.inference_single(_model, image, thresh)
        _, img_encoded = cv2.imencode('.jpg', image)
        return send_file(io.BytesIO(img_encoded), as_attachment=True, attachment_filename='image_detected.jpg',
                         mimetype='image/jpg')
    except KeyError as e:
        return make_response(('There is something wrong with the payload.', 400))
    except ValueError as e:
        return make_response(('There is something wrong with the payload.', 400))
    return make_response(('Something went wrong.', 500))


if __name__ == '__main__':
    app.run(debug=False, threaded=False)
