from flask import current_app
from datetime import timedelta
import io
from PIL import Image

def get_config(key, default_value):
    return int(current_app.config.get(key, default_value))

def get_jwt_expiration():
    expires_in = get_config('JWT_EXPIRES_IN', 1)
    return timedelta(seconds=expires_in)
