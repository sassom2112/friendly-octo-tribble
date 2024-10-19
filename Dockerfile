# Use an official AWS Lambda Python runtime as a base image
FROM public.ecr.aws/lambda/python:3.8

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy your application code
COPY src/app.py ./
COPY src/model.pth ./

# Set the CMD to your Lambda handler (app.lambda_handler)
CMD ["app.lambda_handler"]
