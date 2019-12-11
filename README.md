# Object detection API using YOLOv3, Flask and Swagger

## About
This project uses a pre-trained yoloV3 object detection model on the COCO dataset.
The bounding boxes and the model translation to Keras was strongly inspired by https://github.com/experiencor/keras-yolo3
Please feel free to contribute or let me know about any issues.

Official yoloV3 website:
https://pjreddie.com/darknet/yolo/

## Build and run in production
Use ``docker build --tag yolov3 .`` in the project directory to build the docker image.
It will fetch and install the necessary dependencies inside the docker container.
Next, run the image using ``docker run -p 80:8000 yolov3``
The Swagger UI is accessible under ``localhost/ui``.
Fell free to use some of the example images from the `images` directory.

## Run in development environment
You can run the application in development mode by following these steps.
First, set up a Python virtual environment by executing ``python3 -m venv venv && source venv/bin/activate`` in the project directory. 
Next, fetch and install the dependencies ``pip3 install -r requirements.txt``
At this point you should be able to start the Flask server by running ``cd api && python api/app.py``
In order to you the hot-reload for development you might change the `threaded` argument to `True`. 
However, do not forget to switch it back to `False` since the Tensorflow model does not work well with the threaded flag set.
The development server is running on `localhost:8080` and the Swagger UI on `localhost:8080/ui`
 
## Swagger UI
The Swagger UI allows for a convenient interaction with the API.
Use the `api/v1/boxes` endpoint to draw boxes around the detected objects or `api/v1/classify` for classification.
In both cases simply upload an image from your local filesystem and click on `Execute`.
Depending on your machine it could take up to 10-15 seconds to process the image.


## License
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
