import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="yolov3_flask",
    author="Adrian Gruszczynski",
    author_email="gruszczynski.adrian@gmail.com",
    description="Object detection API built using YOLOv3 model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/adeaeede/yolov3-flask",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
