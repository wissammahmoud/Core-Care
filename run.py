import os
from app import create_app
from app.services.model_service import ModelManager
from app.services.milvus_service import MilvusManager

# Initialize services on startup
def initialize_services():
    """Pre-load models and connect to services"""
    print("Initializing services...")
    
    # Connect to Milvus
    milvus = MilvusManager.get_instance()
    milvus.connect(
        host=os.getenv('MILVUS_HOST', 'localhost'),
        port=int(os.getenv('MILVUS_PORT', '19530')),
        collection_name=os.getenv('MILVUS_COLLECTION', 'food_nutrition2')
    )
    
    # Optionally pre-load model (can also lazy load on first request)
    # model_mgr = ModelManager.get_instance()
    # model_mgr.load_model(os.getenv('MODEL_ID'))
    
    print("Services initialized!")

if __name__ == '__main__':
    # app = create_app(os.getenv('FLASK_ENV', 'development'))
    from flask import Flask
    from app.controllers.routes import api_bp 
    from werkzeug.middleware.proxy_fix import ProxyFix


    app = Flask(__name__)
    app.register_blueprint(api_bp) 
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

    # Initialize services before starting
    initialize_services()
    
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000)),
        # debug=app.config['DEBUG'],
        debug= False,
        use_reloader=True
    )