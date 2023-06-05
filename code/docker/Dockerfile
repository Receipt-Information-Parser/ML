FROM public.ecr.aws/lambda/python:3.9

COPY app.py requirements.txt ./
COPY model /opt/ml/model
COPY process /opt/ml/process

RUN python -m pip install --upgrade pip

RUN python3.9 -m pip install -r requirements.txt -t .
RUN python3.9 -m pip install --upgrade urllib3


CMD ["app.lambda_handler"]

