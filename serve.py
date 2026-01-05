#!/usr/bin/env python3
"""
Serve Resume Classifier API

Simple script to start the API server. Just run it.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.api_server import create_app


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("STARTING API SERVER")
    print("=" * 70)
    print("Endpoints:")
    print("  GET  http://localhost:5000/health")
    print("  GET  http://localhost:5000/info")
    print("  POST http://localhost:5000/predict")
    print("  POST http://localhost:5000/predict_batch")
    print("\nPress Ctrl+C to stop")
    print("=" * 70 + "\n")

    try:
        app = create_app()
        app.run(host="0.0.0.0", port=5000)
    except KeyboardInterrupt:
        print("\n\n✓ Server stopped\n")
