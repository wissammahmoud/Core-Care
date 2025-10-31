"""Image processing utilities"""
from PIL import Image
import io
from typing import Union

def load_and_resize_image(
    image_source: Union[bytes, str, Image.Image],
    max_size: int = 448
) -> Image.Image:
    """
    Load image from various sources and resize if needed
    
    Args:
        image_source: bytes, file path, or PIL Image
        max_size: Maximum dimension (width or height)
    
    Returns:
        PIL Image in RGB format
    """
    # Convert to PIL Image
    if isinstance(image_source, Image.Image):
        img = image_source
    elif isinstance(image_source, bytes):
        img = Image.open(io.BytesIO(image_source))
    else:
        img = Image.open(image_source)
    
    # Convert to RGB
    img = img.convert("RGB")
    
    # Resize if needed
    if img.width > max_size or img.height > max_size:
        ratio = min(max_size / img.width, max_size / img.height)
        new_width = int(img.width * ratio)
        new_height = int(img.height * ratio)
        
        img = img.resize(
            (new_width, new_height),
            Image.Resampling.LANCZOS
        )
    
    return img

def validate_image_dimensions(img: Image.Image) -> tuple[bool, str]:
    """
    Validate image dimensions
    
    Returns:
        (is_valid, error_message)
    """
    if img.width < 50 or img.height < 50:
        return False, "Image too small (min 50x50 pixels)"
    
    if img.width > 4000 or img.height > 4000:
        return False, "Image too large (max 4000x4000 pixels)"
    
    return True, ""