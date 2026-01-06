FROM python
WORKDIR /app
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the FastAPI port
EXPOSE 8000

# Command to run FastAPI (recommended form)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]