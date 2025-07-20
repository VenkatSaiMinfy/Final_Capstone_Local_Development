# src/app/__init__.py

from flask import Flask
import os

def create_app():
    """
    Factory to create and configure the Flask application.
    
    - Sets up template and static folders.
    - Loads SECRET_KEY from environment (falls back to 'dev-key').
    - Registers the routes blueprint.
    
    Returns:
        Flask app instance
    """
    # ─────────────────────────────────────────────
    # Initialize Flask app with custom folders
    # ─────────────────────────────────────────────
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static"
    )
    
    # ─────────────────────────────────────────────
    # Configure secret key for session signing
    # ─────────────────────────────────────────────
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-key")
    
    # ─────────────────────────────────────────────
    # Register routes blueprint from src/app/routes.py
    # ─────────────────────────────────────────────
    from app.routes import bp as routes_bp
    app.register_blueprint(routes_bp)
    
    return app
