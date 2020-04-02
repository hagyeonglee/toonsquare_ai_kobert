FROM pytorch/pytorch:1.2-cuda10.0-cudnn7-devel
USER root
MAINTAINER lhky0708@gmail.com
RUN pip install pytorch_transformers flask pandas
COPY requirements.txt /tmp
WORKDIR /tmp
RUN pip install --trusted-host pypi.python.org -r requirements.txt
RUN pip install -U protobuf
RUN pip install -U flask-cors
#RUN pip install git -y
RUN git clone https://github.com/NVIDIA/apex.git && cd apex && python setup.py install --cuda_ext --cpp_ext
#RUN git config --global url."https://83641c24057c7cc1e8c39c2a1460c66819cdbf32@github.com/".insteadOf "https://github.com/"
RUN mkdir /app
RUN git clone "https://github.com/lhky78/toonsquare_ai_kobert.git" /app
RUN cd /app && git fetch
EXPOSE 3000 3000
#CMD cd /app/kobert && git pull --rebase && python app_azure_test.py
CMD cd /app/kobert && git pull --rebase && python app_translator_test.py
#CMD ["python", "app_azure_test.py"]


