FROM ubuntu:16.04
FROM python:3
ADD train.py /
ADD data.py /
ADD evaluate.py /
ADD replay.py /
ADD mobilenet.py /
RUN pip3 install torch torchvision
RUN pip3 install numpy
RUN pip3 install pandas
RUN pip3 install Pillow
CMD [ "python3", "train.py"]
