import os
import sys

# Add the research_agent directory to Python path so internal imports work on Vercel
current_dir = os.path.dirname(__file__)
research_agent_dir = os.path.join(current_dir, "..", "research_agent")
sys.path.append(os.path.abspath(research_agent_dir))

# Import the FastAPI app instance for Vercel Serverless Function to serve
from server import app
