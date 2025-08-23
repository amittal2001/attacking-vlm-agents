# Start from the WAA base image
FROM windowsarena/winarena:latest

# Copy your requirements
COPY requirements.txt /tmp/requirements.txt
COPY src/win-arena-container/client /client

# Install extra dependencies
RUN pip install --no-cache-dir -r /tmp/requirements.txt
