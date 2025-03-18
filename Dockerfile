FROM python:3.9

# Step 1: Switch to root user to install dependencies
USER root

# Ensure /var/lib/apt/lists exists and has the right permissions
RUN mkdir -p /var/lib/apt/lists/partial && \
    chmod -R 755 /var/lib/apt/lists

# Step 2: Install necessary packages
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    apt-get install -y libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

# Step 3: Create the user
RUN adduser --disabled-password --gecos "" myuser

# Step 4: Create the /uploads directory and set ownership
RUN mkdir /uploads && chown -R myuser:myuser /uploads

# Step 5: Switch to myuser to run the app
USER myuser
ENV PATH="/home/myuser/.local/bin:$PATH"

# Step 6: Set the working directory
WORKDIR /app

# Step 7: Copy requirements file with proper ownership
COPY --chown=myuser ./requirements.txt requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade -r requirements.txt
RUN zenml login https://ntchinda-giscard-zenml.hf.space --api-key ZENKEY_eyJpZCI6IjZhY2M1Y2QyLThlMmEtNGRhZC05ODZmLTgyZmFmMmMxNDdjYSIsImtleSI6IjNkZmIxYjFkMzc1NDQ0ZGQyMzdhMThjYzhhZGE2YWNkZDM0YjVmNzk5MWIyOWRjZjcyMmFlMzYzY2E2ZTJiNDkifQ==


# Step 8: Copy the rest of the application code with proper ownership
COPY --chown=myuser . /app

# Step 9: Command to run the application
CMD ["python", "main.py"]
CMD ["streamlit", "run", "app.py", "--server.port", "7860"]