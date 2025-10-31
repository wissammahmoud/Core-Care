"""
Enhanced parsers for nutrition analysis with robust JSON extraction.
"""
import json
import re
from typing import Dict
import logging

logger = logging.getLogger(__name__)


def extract_json_from_text(text: str) -> Dict:
    """
    Extract and parse JSON from text with multiple fallback strategies.
    
    Args:
        text: Raw text that may contain JSON
    
    Returns:
        Parsed JSON dict
    
    Raises:
        ValueError: If no valid JSON can be extracted
    """
    # Strategy 1: Try direct JSON parse
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError as e:
        logger.debug(f"Direct JSON parse failed: {e}")
    
    # Strategy 2: Extract JSON from code blocks
    # Look for ```json ... ``` or ```...```
    json_block_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
    matches = re.findall(json_block_pattern, text, re.DOTALL)
    if matches:
        try:
            return json.loads(matches[0])
        except json.JSONDecodeError as e:
            logger.debug(f"Code block JSON parse failed: {e}")
    
    # Strategy 3: Extract first JSON object from text
    # Look for { ... } anywhere in the text
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    
    # Strategy 4: Try to convert Python-like format to JSON
    # Handle format like: ['ingredient:weight', 'ingredient:weight']
    try:
        result = convert_python_format_to_json(text)
        if result:
            return result
    except Exception as e:
        logger.debug(f"Python format conversion failed: {e}")
    
    # Strategy 5: Try to extract from structured text
    try:
        result = extract_from_structured_text(text)
        if result:
            return result
    except Exception as e:
        logger.debug(f"Structured text extraction failed: {e}")
    
    # All strategies failed
    raise ValueError(f"Could not extract valid JSON from text. Original: {text[:200]}")


def convert_python_format_to_json(text: str) -> Dict:
    """
    Convert Python list format to JSON.
    
    Example input: ['burger:250g', 'fries:150g']
    Example output: {"portions": {"burger": 250, "fries": 150}}
    """
    # Find list-like patterns
    list_pattern = r"\[([^\]]+)\]"
    matches = re.findall(list_pattern, text)
    
    if not matches:
        return None
    
    portions = {}
    
    for match in matches:
        # Split by comma and process each item
        items = match.split(',')
        for item in items:
            item = item.strip().strip("'\"")
            
            # Try to split by colon
            if ':' in item:
                parts = item.split(':', 1)
                ingredient = parts[0].strip()
                weight_str = parts[1].strip()
                
                # Extract number from weight string (e.g., "250g" -> 250)
                weight_match = re.search(r'(\d+)', weight_str)
                if weight_match:
                    weight = int(weight_match.group(1))
                    portions[ingredient] = weight
    
    if portions:
        return {"portions": portions}
    
    return None


def extract_from_structured_text(text: str) -> Dict:
    """
    Extract portions from structured text output.
    
    Example input:
    Dish: Chicken Burger
    Portion Size: ['burger:250g', 'fries:150g']
    
    Example output: {"portions": {"burger": 250, "fries": 150}}
    """
    portions = {}
    
    # Look for "Portion Size:" or "Portions:" line
    portion_patterns = [
        r"Portion\s*Size\s*:\s*(.+)",
        r"Portions\s*:\s*(.+)",
        r"Ingredients.*:\s*\[([^\]]+)\]",
    ]
    
    for pattern in portion_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            portion_text = match.group(1)
            
            # Try to parse as Python list
            if '[' in portion_text:
                result = convert_python_format_to_json(portion_text)
                if result:
                    return result
    
    # Look for ingredient:weight patterns anywhere
    ingredient_pattern = r"['\"]?([a-zA-Z\s]+)['\"]?\s*:\s*['\"]?(\d+)g?['\"]?"
    matches = re.findall(ingredient_pattern, text)
    
    for ingredient, weight in matches:
        ingredient = ingredient.strip()
        weight = int(weight)
        if ingredient and weight > 0:
            portions[ingredient] = weight
    
    if portions:
        return {"portions": portions}
    
    return None


def extract_portions_from_json(parsed_json: Dict) -> Dict[str, float]:
    """
    Extract ingredient portions from parsed JSON.
    
    Handles various JSON formats:
    - {"portions": {"chicken": 150}}
    - {"portion_size": [{"name": "chicken", "grams": 150}]}
    - {"ingredients": [{"ingredient": "chicken", "weight": 150}]}
    
    Args:
        parsed_json: Parsed JSON dict
    
    Returns:
        Dict mapping ingredient name to grams
    """
    portions = {}
    
    # Format 1: Direct portions dict
    if "portions" in parsed_json:
        portions_data = parsed_json["portions"]
        if isinstance(portions_data, dict):
            # {"portions": {"chicken": 150}}
            for ingredient, weight in portions_data.items():
                try:
                    portions[ingredient] = float(weight)
                except (ValueError, TypeError):
                    logger.warning(f"Invalid weight for {ingredient}: {weight}")
        elif isinstance(portions_data, list):
            # {"portions": [{"name": "chicken", "grams": 150}]}
            for item in portions_data:
                if isinstance(item, dict):
                    name = item.get("name") or item.get("ingredient")
                    weight = item.get("grams") or item.get("weight") or item.get("amount")
                    if name and weight:
                        try:
                            portions[name] = float(weight)
                        except (ValueError, TypeError):
                            pass
    
    # Format 2: portion_size list
    elif "portion_size" in parsed_json:
        portion_size = parsed_json["portion_size"]
        if isinstance(portion_size, list):
            for item in portion_size:
                if isinstance(item, dict):
                    name = item.get("name") or item.get("ingredient")
                    weight = item.get("grams") or item.get("weight") or item.get("amount")
                    if name and weight:
                        try:
                            portions[name] = float(weight)
                        except (ValueError, TypeError):
                            pass
                elif isinstance(item, str):
                    # Format: "ingredient:weight"
                    if ':' in item:
                        parts = item.split(':', 1)
                        ingredient = parts[0].strip()
                        weight_str = parts[1].strip().rstrip('g')
                        try:
                            portions[ingredient] = float(weight_str)
                        except ValueError:
                            pass
    
    # Format 3: ingredients list
    elif "ingredients" in parsed_json:
        ingredients = parsed_json["ingredients"]
        if isinstance(ingredients, list):
            for item in ingredients:
                if isinstance(item, dict):
                    name = item.get("name") or item.get("ingredient")
                    weight = item.get("grams") or item.get("weight") or item.get("amount")
                    if name and weight:
                        try:
                            portions[name] = float(weight)
                        except (ValueError, TypeError):
                            pass
    
    return portions


def validate_json_output(text: str) -> bool:
    """
    Quick check if text is valid JSON.
    
    Args:
        text: Text to check
    
    Returns:
        True if valid JSON, False otherwise
    """
    try:
        json.loads(text.strip())
        return True
    except json.JSONDecodeError:
        return False