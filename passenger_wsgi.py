import os
import sys

# Add the application path
sys.path.insert(0, os.path.dirname(__file__))

# Import your FastAPI app from main.py
from main import app as f_app

# Convert ASGI (FastAPI) to WSGI since cPanel/Passenger needs WSGI
from a2wsgi import ASGIMiddleware

# Passenger expects the entry point to be named "application"
application = ASGIMiddleware(f_app)
