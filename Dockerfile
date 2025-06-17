# Dockerfile for FastAPI

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI application code
COPY . .

# Copy the .env file to the working directory
# Ensure .env is inside /app so dotenv can find it

COPY .env .env

# Expose the port FastAPI will run on
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
