# Activate virtual environment if it exists
if [ -d "antenv" ]; then
    source antenv/bin/activate
fi

# Install dependencies
pip install -r requirements.txt

# Start the application with uvicorn directly
uvicorn app:app --host 0.0.0.0 --port 8000