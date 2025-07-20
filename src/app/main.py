#!/usr/bin/env python3
# src/app/main.py

import os
import sys

# ────────────────────────────────────────────────────────────────
# 1) Ensure project `src/` directory is on Python import path
# ────────────────────────────────────────────────────────────────
here = os.path.dirname(__file__)                                  # .../lead_scoring_project/src/app
project_src = os.path.abspath(os.path.join(here, ".."))           # .../lead_scoring_project/src
if project_src not in sys.path:
    sys.path.insert(0, project_src)

# ────────────────────────────────────────────────────────────────
# 2) Import the application factory
# ────────────────────────────────────────────────────────────────
from app import create_app

# ────────────────────────────────────────────────────────────────
# 3) Create Flask application instance
# ────────────────────────────────────────────────────────────────
app = create_app()

# ────────────────────────────────────────────────────────────────
# 4) Run the application when executed directly
#    - host="0.0.0.0" binds to all interfaces
#    - port=5001 (customizable)
#    - debug=True for local development only!
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
