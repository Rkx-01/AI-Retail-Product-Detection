FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user

# Switch to the "user" user
USER user

# Set home to the user's home directory
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    FLASK_APP=app.py

# Set the working directory to the user's home directory
WORKDIR $HOME/app

# Install dependencies FIRST (caching)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# PRE-DOWNLOAD AI WEIGHTS AS THE APP USER
# This ensures they are saved in /home/user/.cache/ and readable at runtime
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
RUN python -c "import torch; from torchvision.models import mobilenet_v2, MobileNet_V2_Weights; mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)"

# Copy the rest of the application files with user ownership
COPY --chown=user . .

# Create dynamic folders
RUN mkdir -p dataset/samples static outputs uploads

EXPOSE 8001

CMD ["gunicorn", "-b", "0.0.0.0:8001", "-w", "1", "--threads", "4", "--timeout", "120", "app:app"]
