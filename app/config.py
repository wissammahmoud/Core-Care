import os
from pathlib import Path

class Config:
    # Flask
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key')
    DEBUG = os.getenv('FLASK_DEBUG', 'False') == 'True'
    
    # Model
    MODEL_ID = os.getenv('MODEL_ID', 'unsloth/Qwen3-VL-4B-Instruct-bnb-4bit')
    MAX_IMAGE_SIZE = int(os.getenv('MAX_IMAGE_SIZE', '448'))
    MAX_NEW_TOKENS = int(os.getenv('MAX_NEW_TOKENS', '256'))
    
    # Milvus
    MILVUS_HOST = os.getenv('MILVUS_HOST', 'localhost')
    MILVUS_PORT = int(os.getenv('MILVUS_PORT', '19530'))
    MILVUS_COLLECTION = os.getenv('MILVUS_COLLECTION', 'nutrition_db')
    
    # Timeouts
    INFERENCE_TIMEOUT = int(os.getenv('INFERENCE_TIMEOUT', '30'))
    REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', '60'))
    
    # Limits
    MAX_FILE_SIZE = int(os.getenv('MAX_FILE_SIZE', '10')) * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}