from . import bound_box

anchors = [[116, 90, 156, 198, 373, 326], [30, 61, 62, 45, 59, 119], [10, 13, 16, 30, 33, 23]]
labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
          "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
          "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
          "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
          "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
          "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
          "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
          "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
          "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
          "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]


def draw_boxes(image, yhat, obj_thresh, nms_thresh=.45):
    input_w, input_h = 416, 416
    image_h, image_w, _ = image.shape

    boxes = list()
    for i in range(len(yhat)):
        # decode the output of the network
        boxes += bound_box.decode_netout(yhat[i][0], anchors[i], obj_thresh, nms_thresh, input_h, input_w)
    bound_box.correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)
    bound_box.do_nms(boxes, nms_thresh)
    probabilities = get_labels(boxes, obj_thresh)
    image = bound_box.draw_boxes(image, boxes, labels, obj_thresh)
    return probabilities, image


def get_labels(boxes, obj_thresh, nms_thresh=.45):
    classes = list()
    for box in boxes:
        for i in range(len(labels)):
            if box.classes[i] > obj_thresh:
                classes.append(f'{labels[i]}: {str(box.classes[i] * 100)}%')
    return classes
