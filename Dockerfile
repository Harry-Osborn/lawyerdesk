# Use Python 3.9 base image
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy your Python script
COPY main.py .

# Run the Python script
CMD ["python", "main.py"]


EXPOSE 8000