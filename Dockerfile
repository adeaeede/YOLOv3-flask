FROM python:3.6
ADD api api
ADD lib lib
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir
WORKDIR /api
EXPOSE 8000
#CMD ["cd", "app/api"]
CMD ["gunicorn", "-b", "0.0.0.0:8000", "app"]
