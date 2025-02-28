# Use the official Chroma DB image as a base
FROM chromadb/chroma:0.6.3

# Create a non-root user with UID 1000 (required by Hugging Face Spaces)
RUN useradd -m -u 1000 appuser
USER appuser

# Set environment variables for persistence and telemetry (optional)
ENV IS_PERSISTENT=TRUE
ENV PERSIST_DIRECTORY=/chroma/chroma
ENV ANONYMIZED_TELEMETRY=TRUE

# Expose the port that Chroma DB will listen on
EXPOSE 8000

# Launch the Chroma DB server on all interfaces at port 8000
CMD ["chromadb", "serve", "--host", "0.0.0.0", "--port", "8000"]