FROM pytorch/pytorch:1.2-cuda10.0-cudnn7-devel
USER root
MAINTAINER lhky0708@gmail.com
ENV LANG ko_KR.utf8
ENV LANGUAGE ko_KR.utf8
ENV LC_ALL ko_KR.utf8
RUN pip install --trusted-host pypi.python.org -r requirements.txt
RUN pip install pytorch_transformers flask pandas
RUN pip install -U protobuf
RUN git clone https://github.com/NVIDIA/apex.git && cd apex && python setup.py install --cuda_ext --cpp_ext
RUN git config --global url."https://83641c24057c7cc1e8c39c2a1460c66819cdbf32@github.com/".insteadOf "https://github.com/"
RUN cd /root/deploy/toonsquare_ai_kobert && git fetch
EXPOSE 3000 3000
CMD cd /root/deploy/toonsquare_ai_kobert/kobert && git pull --rebase && python app_azure_test.py
#CMD ["python", "app_azure_test.py"]
