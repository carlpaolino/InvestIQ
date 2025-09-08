#!/usr/bin/env python3
"""
Simple script to start the InvestIQ API server.
"""

import uvicorn
from investiq.prediction.api import app

if __name__ == "__main__":
    print("Starting InvestIQ API server...")
    print("Web UI will be available at: http://localhost:8000/ui")
    print("API docs will be available at: http://localhost:8000/docs")
    print("Press Ctrl+C to stop the server")
    
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info"
    )
