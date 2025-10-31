"""Input validation utilities"""
from werkzeug.datastructures import FileStorage
import os

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'gif'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_image_upload(file: FileStorage) -> tuple[bool, str]:
    """
    Validate uploaded image file
    
    Returns:
        (is_valid, error_message)
    """
    if not file:
        return False, "No file provided"
    
    if file.filename == '':
        return False, "Empty filename"
    
    if not allowed_file(file.filename):
        return False, f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
    
    # Check file size (if available)
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)  # Reset position
    
    if file_size > MAX_FILE_SIZE:
        return False, f"File too large. Max size: {MAX_FILE_SIZE // 1024 // 1024}MB"
    
    return True, ""

def validate_ingredients_list(ingredients: list) -> tuple[bool, str]:
    """Validate ingredients list"""
    if not isinstance(ingredients, list):
        return False, "Ingredients must be a list"
    
    if len(ingredients) == 0:
        return False, "Ingredients list cannot be empty"
    
    if len(ingredients) > 50:
        return False, "Too many ingredients (max 50)"
    
    for ingredient in ingredients:
        if not isinstance(ingredient, str):
            return False, "Each ingredient must be a string"
        
        if len(ingredient.strip()) == 0:
            return False, "Empty ingredient name"
        
        if len(ingredient) > 100:
            return False, "Ingredient name too long (max 100 chars)"
    
    return True, ""
