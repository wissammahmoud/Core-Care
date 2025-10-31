from flask import Flask
from flask_cors import CORS
import logging
from app.config import config

def create_app(config_name='default'):
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    
    # CORS
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    
    # Logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Register blueprints
    from app.controllers.routes import api_bp
    app.register_blueprint(api_bp, url_prefix='/api')
    
    # Error handlers
    @app.errorhandler(413)
    def request_entity_too_large(error):
        return {"error": "File too large. Max size: 10MB"}, 413
    
    @app.errorhandler(500)
    def internal_error(error):
        return {"error": "Internal server error"}, 500
    
    return app