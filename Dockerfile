# Start from the WAA base image
FROM windowsarena/winarena:latest

# Copy your requirements
COPY requirements.txt /tmp/requirements.txt

# Install extra dependencies
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# copy src code
COPY src/win-arena-container/client /client
