from lib import bound_box
from lib.draw_boxes import draw_boxes as db

# define the expected input shape for the model
INPUT_W, INPUT_H = 416, 416


def inference_single(model, image, thresh):
    new_image = bound_box.preprocess_input(image, INPUT_W, INPUT_H)
    y_hat = model.predict(new_image)
    return db(image, y_hat, obj_thresh=thresh)
