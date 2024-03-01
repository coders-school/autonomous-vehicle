FROM python:3.11

# Install libGL.so
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file to the container
COPY python/requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code to the container
COPY . .

# Start from python/segmentation directory
WORKDIR /app/python/segmentation

# Set the entry point for the container
CMD [ "python", "main.py" ]
