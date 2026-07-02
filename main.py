import sys
import os

# Add research_agent to path so Railway can find server.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "research_agent"))

from server import app
