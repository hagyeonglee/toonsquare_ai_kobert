FROM pytorch/pytorch:1.2-cuda10.0-cudnn7-devel
WORKDIR /app
ADD . /app
RUN pip install --trusted-host pypi.python.org -r requirements.txt
RUN pip install pytorch_transformers flask pandas
RUN pip install -U protobuf
RUN git clone https://github.com/NVIDIA/apex.git && cd apex && python setup.py install --cuda_ext --cpp_ext
EXPOSE 80
CMD ["python", "app_test.py"]

