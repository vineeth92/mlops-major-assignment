# Use official Python image
FROM python:3.11

# Set working directory inside container
WORKDIR /app

# Copy all files from local dir to container's /app
COPY . .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Run prediction script (you can change this to train or quantize if needed)
CMD ["python", "src/predict.py"]
