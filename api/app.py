from flask import Flask
from flask import request, Response, send_file
from keras.models import model_from_json
from lib import inference
import cv2
import io
import base64
import numpy as np
import json
from flask_swagger_ui import get_swaggerui_blueprint

SWAGGER_URL = '/ui'
API_URL = '/static/swagger.json'
SWAGGERUI_BLUEPRINT = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "YOLOv3 object detection API"
    }
)

application = Flask(__name__)
application.register_blueprint(SWAGGERUI_BLUEPRINT, url_prefix=SWAGGER_URL)


def init_model():
    # load json and create model
    json_file = open('static/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("static/model.h5")
    print("Loaded model from disk")
    return model


# Provide the global model object for image recognition
_model = init_model()


def encode_image(_bytes):
    ndarray = np.fromstring(_bytes, np.uint8)
    return cv2.imdecode(ndarray, cv2.IMREAD_COLOR)


@application.route('/api/v1/boxes/', methods=['POST'])
def boxes():
    if request.content_length > 4194304:
        return Response(json.dumps({'error': 'Exceeded maximal content size'}), status=413,
                        mimetype='application/json')
    try:
        _bytes = request.get_data()
        image = encode_image(_bytes)
        probabilities, image = inference.inference_single(_model, image, thresh=0.5)
        _, img_encoded = cv2.imencode('.jpg', image)
        return send_file(io.BytesIO(img_encoded), as_attachment=True, attachment_filename='image_detected.jpg',
                         mimetype='image/jpg', )
    except KeyError as e:
        print(e)
        return Response(json.dumps({'error': 'Payload does not comply the specification.'}), status=400,
                        mimetype='application/json')
    except ValueError as e:
        print(e)
    return Response(json.dumps({'error': 'Payload does not comply the specification.'}), status=400,
                    mimetype='application/json')


@application.route('/api/v1/classify/', methods=['POST'])
def classify():
    if request.content_length > 4194304:
        return Response(json.dumps({'error': 'Exceeded maximal content size'}), status=413,
                        mimetype='application/json')
    try:
        _bytes = request.get_data()
        image = encode_image(_bytes)
        probabilities, image = inference.inference_single(_model, image, thresh=0.1)
        return Response(json.dumps({'classes': probabilities}), status=200, mimetype='application/json')
    except KeyError as e:
        return Response(json.dumps({'error': 'Payload does not comply the specification.'}), status=400,
                        mimetype='application/json')
    except ValueError as e:
        return Response(json.dumps({'error': 'Payload does not comply the specification.'}), status=400,
                        mimetype='application/json')


@application.route('/api/v1/encode', methods=['POST'])
def encode():
    if str(request.content_type) != 'application/octet-stream':
        return Response(json.dumps({'error': 'Unsupported content type. Try using content-type: image/png '}),
                        status=415,
                        mimetype='application/json')
    if request.content_length < 0:
        return Response(json.dumps({'error': 'Exceeded maximal content size'}), status=413,
                        mimetype='application/json')
    image = base64.b64encode(request.get_data()).decode('utf-8')
    return Response(json.dumps({'image': image}), status=200, mimetype='application/json')


if __name__ == '__main__':
    application.run(debug=False, threaded=False, host='localhost', port='8080')
