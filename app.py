from flask import Flask
from flask import jsonify
from keras.models import model_from_json
from yolov3 import inference


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
    print(type(_model))
    probabilities, image = inference.inference_single(_model)
    return jsonify(probabilities)


if __name__ == '__main__':
    app.run(debug=False, threaded=False)
