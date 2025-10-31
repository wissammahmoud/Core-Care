"""
LangGraph Workflows - FULLY FIXED Version with Ingredient Breakdown
===================================================================
FIXES:
1. ✅ All portions converted to grams (ml → g, oz → g, etc.)
2. ✅ Total calories properly calculated from portions × database nutrition data
3. ✅ Supports both image and text-only inputs
4. ✅ Conversational AI with nutrition breakdown
5. ✅ Auto-fixes composite dishes (e.g., "pastries" → individual ingredients)
"""
from typing import TypedDict, Dict, Any, Optional, List, Tuple
from langgraph.graph import StateGraph, END
from PIL import Image
import io
import logging
import re

logger = logging.getLogger(__name__)


# ============================================================================
# COMPOSITE DISH BREAKDOWN UTILITIES
# ============================================================================

def detect_composite_dish(ingredient_name: str, portions: Dict[str, float]) -> bool:
    """Detect if an ingredient name is actually a composite dish."""
    composite_keywords = [
        'pastries', 'pastry', 'burger', 'sandwich', 'pizza', 'taco', 
        'burrito', 'wrap', 'kebab', 'cake', 'pie', 'cookies', 'muffin',
        'donut', 'croissant', 'bagel'
    ]
    
    ingredient_lower = ingredient_name.lower().replace('_', ' ')
    for keyword in composite_keywords:
        if keyword in ingredient_lower:
            logger.warning(f"⚠️ Detected composite dish: '{ingredient_name}'")
            return True
    
    # Single ingredient with large portion often indicates composite dish
    if len(portions) == 1 and portions[ingredient_name] > 100:
        logger.warning(f"⚠️ Single ingredient {portions[ingredient_name]}g - likely composite")
        return True
    
    return False


def suggest_ingredient_breakdown(dish_name: str, total_grams: float, 
                                 ingredients_list: List[str]) -> Dict[str, float]:
    """Suggest breakdown based on typical ingredient ratios."""
    logger.info(f"📊 Breaking down '{dish_name}' ({total_grams}g) into ingredients")
    
    if not ingredients_list:
        return {dish_name: total_grams}
    
    # Typical ingredient weight ratios
    weights = {
        'flour': 3.0, 'rice': 3.0, 'pasta': 3.0, 'dough': 3.0,
        'chicken': 2.5, 'beef': 2.5, 'pork': 2.5, 'fish': 2.5,
        'eggs': 1.5, 'cheese': 1.0,
        'butter': 0.5, 'oil': 0.3, 'olive_oil': 0.3,
        'sugar': 0.8, 'honey': 0.5,
        'pistachios': 0.7, 'almonds': 0.7, 'walnuts': 0.7, 'nuts': 0.7,
        'lettuce': 0.5, 'tomato': 0.8, 'onion': 0.6,
        'ketchup': 0.3, 'mayo': 0.3, 'mustard': 0.2
    }
    
    # Assign weights
    ingredient_units = {}
    total_weight_units = 0
    
    for ingredient in ingredients_list:
        ing_lower = ingredient.lower().replace('_', ' ')
        weight = 1.0  # default
        for key, value in weights.items():
            if key in ing_lower:
                weight = value
                break
        ingredient_units[ingredient] = weight
        total_weight_units += weight
    
    # Distribute proportionally
    breakdown = {}
    for ingredient, units in ingredient_units.items():
        proportion = units / total_weight_units
        grams = total_grams * proportion
        breakdown[ingredient] = round(grams, 1)
        logger.info(f"   {ingredient}: {grams:.1f}g ({proportion*100:.1f}%)")
    
    return breakdown


def fix_composite_dish_portions(parsed_json: Dict, portions: Dict[str, float]) -> Dict[str, float]:
    """Fix portions if model returned composite dish instead of breakdown."""
    logger.info("🔍 Checking for composite dishes...")
    
    ingredients_list = parsed_json.get("ingredients", [])
    composite_dishes = []
    
    for ingredient_name, grams in portions.items():
        if detect_composite_dish(ingredient_name, portions):
            composite_dishes.append((ingredient_name, grams))
    
    if not composite_dishes:
        logger.info("✅ No composite dishes - portions are good!")
        return portions
    
    # Fix each composite dish
    fixed_portions = {}
    for dish_name, total_grams in composite_dishes:
        logger.warning(f"🔧 Fixing: '{dish_name}' → individual ingredients")
        breakdown = suggest_ingredient_breakdown(dish_name, total_grams, ingredients_list)
        fixed_portions.update(breakdown)
    
    # Keep non-composite ingredients
    for ingredient_name, grams in portions.items():
        if ingredient_name not in [d[0] for d in composite_dishes]:
            fixed_portions[ingredient_name] = grams
    
    logger.info(f"✅ Fixed to {len(fixed_portions)} individual ingredients")
    return fixed_portions


# ============================================================================
# STATE SCHEMA
# ============================================================================

class NutritionWorkflowState(TypedDict, total=False):
    """State for the nutrition analysis workflow."""
    # Input fields
    image: Optional[bytes]
    instruction: str
    system_prompt: Optional[str]
    
    # Vision analysis output
    raw_text: str
    parsed_json: Optional[Dict[str, Any]]
    portions: Optional[Dict[str, float]]  # ALL IN GRAMS
    
    # Database lookup output
    database_results: Optional[List[Dict[str, Any]]]
    milvus_results: Optional[List[Dict[str, Any]]]
    
    # LLM analysis output
    llm_analysis: Optional[Dict[str, Any]]
    
    # Final nutrition result
    nutrition_result: Optional[Dict[str, Any]]
    
    # Error handling
    error: Optional[str]
    error_stage: Optional[str]


# ============================================================================
# UNIT CONVERSION UTILITIES
# ============================================================================

def convert_to_grams(amount: float, unit: str) -> float:
    """
    Convert various units to grams.
    
    Args:
        amount: The numeric amount
        unit: The unit (g, ml, oz, lb, cup, tbsp, tsp, etc.)
    
    Returns:
        Amount in grams
    """
    unit = unit.lower().strip()
    
    # Already in grams
    if unit in ['g', 'gram', 'grams', 'gr']:
        return amount
    
    # Milliliters (assume density of water: 1ml = 1g)
    # For oils: 1ml ≈ 0.92g, but we'll use 1:1 for simplicity
    if unit in ['ml', 'milliliter', 'milliliters', 'mL']:
        return amount
    
    # Liters
    if unit in ['l', 'liter', 'liters', 'L']:
        return amount * 1000
    
    # Ounces (weight)
    if unit in ['oz', 'ounce', 'ounces']:
        return amount * 28.35
    
    # Pounds
    if unit in ['lb', 'lbs', 'pound', 'pounds']:
        return amount * 453.592
    
    # Cups (approximate, varies by ingredient)
    if unit in ['cup', 'cups', 'c']:
        return amount * 240  # Assume liquid cup
    
    # Tablespoons
    if unit in ['tbsp', 'tablespoon', 'tablespoons', 'T']:
        return amount * 15
    
    # Teaspoons
    if unit in ['tsp', 'teaspoon', 'teaspoons', 't']:
        return amount * 5
    
    # Kilograms
    if unit in ['kg', 'kilogram', 'kilograms']:
        return amount * 1000
    
    # If unknown unit, log warning and return original amount
    logger.warning(f"Unknown unit '{unit}', treating as grams")
    return amount


def parse_portion_string(portion_str: str) -> tuple[str, float]:
    """
    Parse a portion string like "chicken_breast:250g" or "olive_oil:15ml"
    and return (ingredient_name, grams).
    
    Args:
        portion_str: String in format "ingredient:amount_unit"
    
    Returns:
        Tuple of (ingredient_name, amount_in_grams)
    """
    try:
        if ':' not in portion_str:
            logger.warning(f"Invalid portion format (no colon): {portion_str}")
            return None, 0.0
        
        # Split into ingredient and amount
        ingredient, amount_str = portion_str.split(':', 1)
        ingredient = ingredient.strip()
        amount_str = amount_str.strip()
        
        # Extract number and unit using regex
        # Matches patterns like: "250g", "15ml", "2.5oz", "1.5 cup"
        match = re.match(r'(\d+(?:\.\d+)?)\s*([a-zA-Z]+)?', amount_str)
        
        if not match:
            logger.warning(f"Could not parse amount from: {amount_str}")
            return ingredient, 100.0  # Default 100g
        
        amount = float(match.group(1))
        unit = match.group(2) if match.group(2) else 'g'  # Default to grams if no unit
        
        # Convert to grams
        grams = convert_to_grams(amount, unit)
        
        logger.info(f"  Parsed: {ingredient} = {amount}{unit} → {grams}g")
        return ingredient, grams
        
    except Exception as e:
        logger.error(f"Error parsing portion '{portion_str}': {e}")
        return None, 0.0


# ============================================================================
# NODE 1: Vision Analysis (supports both image and text-only)
# ============================================================================

def vision_analysis_node(state: NutritionWorkflowState) -> Dict[str, Any]:
    """
    Analyze food using vision model (if image provided) or process text-only query.
    ALL PORTIONS RETURNED IN GRAMS.
    """
    import torch
    import json
    
    try:
        logger.info("🔍 Starting vision analysis...")
        
        # Get input data
        image_bytes = state.get("image")
        instruction = state.get("instruction", "Analyze the ingredients in this food.")
        system_prompt = state.get("system_prompt")
        
        # Force JSON output if not explicitly requesting it
        if "json" not in instruction.lower() and "JSON" not in instruction:
            if image_bytes:
                json_instruction = f"""{instruction}

CRITICAL: Respond with ONLY a valid JSON object. No other text, explanations, or markdown. Start directly with {{ and end with }}.

Use this EXACT format (ALL AMOUNTS MUST INCLUDE UNITS):
{{
  "camera_or_phone_prob": <float 0.00-1.00>,
  "food_prob": <float 0.00-1.00>,
  "dish_name": "<lowercase camel case string>",
  "food_type": "<one of: home_cooked | restaurant_food | packaged | dessert | beverage | baked_goods>",
  "ingredients": ["<lowercase_snake_case strings>"],
  "portion_size": ["<ingredient>:<amount><unit>"],
  "cooking_method": "<one of: frying | baking | grilling | boiling | steaming | roasting | raw | sauteing | stew>"
}}

🚨 CRITICAL RULES FOR PORTION_SIZE:
1. Break down composite dishes into INDIVIDUAL INGREDIENTS
2. Estimate portion for EACH ingredient separately  
3. Do NOT use dish names (e.g., "pastries", "burger", "pizza")
4. Use actual ingredient names from the ingredients list
5. ALWAYS include units (g, ml, oz, cup, tbsp, tsp)

Example 1 - Grilled Chicken Kebab:
{{
  "camera_or_phone_prob": 0.95,
  "food_prob": 0.98,
  "dish_name": "grilledChickenKebab",
  "food_type": "restaurant_food",
  "ingredients": ["chicken_breast", "olive_oil", "lemon", "onion"],
  "portion_size": ["chicken_breast:250g", "olive_oil:15ml", "lemon:20g", "onion:50g"],
  "cooking_method": "grilling"
}}

Example 2 - Pistachio Pastries (BREAK DOWN!):
❌ WRONG: "portion_size": ["pastries:150g"]
✅ CORRECT: "portion_size": ["pistachios:30g", "flour:60g", "sugar:25g", "eggs:25g", "butter:20g"]

{{
  "camera_or_phone_prob": 0.98,
  "food_prob": 0.99,
  "dish_name": "pistachioFilledPastries",
  "food_type": "dessert",
  "ingredients": ["pistachios", "flour", "sugar", "eggs", "butter"],
  "portion_size": ["pistachios:30g", "flour:60g", "sugar:25g", "eggs:25g", "butter:20g"],
  "cooking_method": "baking"
}}

Example 3 - Classic Burger (BREAK DOWN!):
❌ WRONG: "portion_size": ["burger:200g"]
✅ CORRECT: "portion_size": ["beef_patty:150g", "burger_bun:80g", "lettuce:20g", ...]

{{
  "camera_or_phone_prob": 0.92,
  "food_prob": 0.97,
  "dish_name": "classicBurger",
  "food_type": "restaurant_food",
  "ingredients": ["beef_patty", "burger_bun", "lettuce", "tomato", "cheese", "onion"],
  "portion_size": ["beef_patty:150g", "burger_bun:80g", "lettuce:20g", "tomato:30g", "cheese:25g", "onion:15g"],
  "cooking_method": "grilling"
}}"""
                instruction = json_instruction
                logger.info("✅ Added JSON formatting requirement to instruction")
        
        # ===================================================================
        # CASE 1: No image - text-only query
        # ===================================================================
        if not image_bytes:
            logger.info("📝 No image provided - text-only mode")
            logger.info(f"Processing instruction: {instruction}")
            
            from app.services.model_adapter import get_model_and_processor
            result = get_model_and_processor(has_image=False)
            
            if isinstance(result, tuple):
                model, processor = result
                logger.info("✅ Unpacked model and processor from tuple")
            elif isinstance(result, dict):
                model = result.get('model')
                processor = result.get('processor')
                logger.info("✅ Extracted model and processor from dict")
            else:
                raise ValueError(f"Unexpected return type from get_model_and_processor: {type(result)}")
            
            logger.info(f"✅ Model loaded for text-only: {type(model).__name__}")
            
            text_prompt_content = f"""Analyze this food description and extract ingredients with their estimated portions.

Food description: {instruction}

CRITICAL: You MUST respond with ONLY a valid JSON object. Do NOT include any other text, explanations, or markdown.

Use this EXACT format (INCLUDE UNITS):
{{
  "camera_or_phone_prob": <float>,
  "food_prob": <float>,
  "dish_name": "<lowercase camel case>",
  "food_type": "<type>",
  "ingredients": ["<ingredients>"],
  "portion_size": ["<ingredient>:<amount><UNIT>"],
  "cooking_method": "<method>"
}}

🚨 CRITICAL: Break down composite dishes into INDIVIDUAL INGREDIENTS!
- ❌ WRONG: "portion_size": ["pastries:150g"]
- ✅ CORRECT: "portion_size": ["flour:60g", "sugar:25g", "eggs:25g", "butter:20g"]

IMPORTANT: Always include units in portion_size (g, ml, oz, cup, tbsp, tsp, etc.)"""
            
            messages = [
                {
                    "role": "system",
                    "content": system_prompt or "You are a nutrition expert. Extract ingredients and portions from food descriptions."
                },
                {
                    "role": "user",
                    "content": text_prompt_content
                }
            ]
            
            logger.info("🤖 Applying chat template for text-only analysis...")
            text_prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Get device
            if hasattr(model, 'device'):
                device = model.device
            elif hasattr(model, 'model') and hasattr(model.model, 'device'):
                device = model.model.device
            else:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            inputs = processor(
                text=[text_prompt],
                return_tensors="pt",
                padding=True
            ).to(device)
            
            logger.info("🚀 Generating text-only analysis...")
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.5,
                    top_p=0.9
                )
            
            tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
            generated_text = tokenizer.batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )[0]
            
            response_text = generated_text.split("assistant\n")[-1].strip()
            
            del inputs, output_ids
            if device.type == "cuda":
                torch.cuda.empty_cache()
        
        # ===================================================================
        # CASE 2: Image provided - vision analysis
        # ===================================================================
        else:
            logger.info("📦 Converting bytes to PIL Image...")
            image = Image.open(io.BytesIO(image_bytes))
            if image.mode != "RGB":
                image = image.convert("RGB")
            logger.info(f"✅ Image converted: {image.size}, mode: {image.mode}")
            
            from app.services.model_adapter import get_model_and_processor
            result = get_model_and_processor(has_image=True)
            
            if isinstance(result, tuple):
                model, processor = result
                logger.info("✅ Unpacked model and processor from tuple")
            elif isinstance(result, dict):
                model = result.get('model')
                processor = result.get('processor')
                logger.info("✅ Extracted model and processor from dict")
            else:
                raise ValueError(f"Unexpected return type: {type(result)}")
            
            logger.info(f"✅ Model loaded: {type(model).__name__}")
            
            conversation = []
            if system_prompt:
                conversation.append({"role": "system", "content": system_prompt})
            
            conversation.append({
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": instruction}
                ]
            })
            
            logger.info("🤖 Applying chat template...")
            text_prompt = processor.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True
            )
            
            if hasattr(model, 'device'):
                device = model.device
            elif hasattr(model, 'model') and hasattr(model.model, 'device'):
                device = model.model.device
            else:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            logger.info("📝 Processing inputs...")
            inputs = processor(
                text=[text_prompt],
                images=[image],
                return_tensors="pt",
                padding=True
            ).to(device)
            
            logger.info("🚀 Generating vision analysis...")
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=0.5,
                    top_p=0.9
                )
            
            logger.info("📤 Decoding output...")
            generated_text = processor.batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )[0]
            
            response_text = generated_text.split("assistant\n")[-1].strip()
            
            del inputs, output_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # ===================================================================
        # Common processing - Parse JSON and extract portions IN GRAMS
        # ===================================================================
        
        logger.info(f"✅ Generated {len(response_text)} characters")
        logger.info(f"Raw response preview: {response_text[:300]}...")
        
        # Parse JSON from response
        json_str = None
        parsed_json = None
        
        # Try multiple extraction methods
        json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            logger.info("✅ Found JSON in markdown code block")
        
        if not json_str:
            json_match = re.search(r'```\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                logger.info("✅ Found JSON in plain code block")
        
        if not json_str:
            start = response_text.find('{')
            end = response_text.rfind('}')
            if start != -1 and end != -1 and start < end:
                json_str = response_text[start:end+1]
                logger.info("✅ Extracted JSON from { to }")
        
        if not json_str:
            logger.warning("⚠️ No JSON delimiters found, trying entire response")
            json_str = response_text
        
        try:
            parsed_json = json.loads(json_str)
            logger.info("✅ JSON parsed successfully")
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON parsing failed: {e}")
            logger.info("🔧 Attempting JSON cleanup...")
            try:
                cleaned = json_str[json_str.find('{'):]
                cleaned = cleaned[:cleaned.rfind('}')+1]
                cleaned = cleaned.replace('**', '')
                parsed_json = json.loads(cleaned)
                logger.info("✅ JSON parsed after cleanup")
            except:
                logger.error("❌ JSON cleanup failed")
                return {
                    "error": f"Failed to parse JSON: {e}",
                    "error_stage": "vision_analysis",
                    "raw_text": response_text
                }
        
        # =================================================================
        # Extract portions and CONVERT ALL TO GRAMS
        # =================================================================
        logger.info("📊 Parsing portions and converting to grams...")
        portions = {}
        
        portion_size_list = parsed_json.get("portion_size", [])
        if portion_size_list:
            for portion_str in portion_size_list:
                ingredient, grams = parse_portion_string(portion_str)
                if ingredient and grams > 0:
                    portions[ingredient] = grams
        
        # Fallback: use ingredients with default 100g
        if not portions:
            logger.warning("⚠️ No portion_size field, using default 100g per ingredient")
            ingredients_list = parsed_json.get("ingredients", [])
            for ing in ingredients_list:
                portions[ing] = 100.0
                logger.info(f"  - {ing}: 100g (default)")
        
        logger.info(f"✅ Extracted {len(portions)} ingredients (ALL IN GRAMS):")
        for name, grams in portions.items():
            logger.info(f"  - {name}: {grams}g")
        
        # Log additional fields
        camera_prob = parsed_json.get("camera_or_phone_prob", 0.0)
        food_prob = parsed_json.get("food_prob", 0.0)
        food_type = parsed_json.get("food_type", "unknown")
        cooking_method = parsed_json.get("cooking_method", "unknown")
        
        logger.info(f"📸 Camera probability: {camera_prob:.2f}")
        logger.info(f"🍽️  Food probability: {food_prob:.2f}")
        logger.info(f"📋 Food type: {food_type}")
        logger.info(f"🔥 Cooking method: {cooking_method}")
        
        # =================================================================
        # FIX COMPOSITE DISHES - Break down if needed
        # =================================================================
        portions = fix_composite_dish_portions(parsed_json, portions)
        
        # Add fixed portions to parsed_json
        parsed_json["portions"] = portions
        
        return {
            "raw_text": response_text,
            "parsed_json": parsed_json,
            "portions": portions  # ALL IN GRAMS, INDIVIDUAL INGREDIENTS
        }
        
    except Exception as e:
        logger.error(f"❌ Vision analysis failed: {e}", exc_info=True)
        return {
            "error": str(e),
            "error_stage": "vision_analysis"
        }


# ============================================================================
# NODE 2: Database Lookup
# ============================================================================

def database_lookup_node(state: NutritionWorkflowState) -> Dict[str, Any]:
    """Search database for top-3 matches for each ingredient."""
    from app.services.milvus_service import MilvusManager
    
    try:
        logger.info("🔍 Starting database lookup...")
        
        portions = state.get("portions", {})
        
        if not portions:
            logger.warning("No portions found, skipping database lookup")
            return {
                "database_results": [],
                "milvus_results": []
            }
        
        ingredient_names = list(portions.keys())
        logger.info(f"Looking up {len(ingredient_names)} ingredients: {ingredient_names}")
        
        # Clean ingredient names
        cleaned_ingredient_names = [
            name.replace('_', ' ').replace('-', ' ') 
            for name in ingredient_names
        ]
        logger.info(f"Cleaned names for search: {cleaned_ingredient_names}")
        
        milvus = MilvusManager.get_instance()
        
        try:
            results = milvus.search_ingredients(cleaned_ingredient_names, top_k=3)
            logger.info(f"✅ Found {len(results)} ingredient matches")
            
            for result in results:
                query = result.get("query", "Unknown")
                matches = result.get("results", [])
                logger.info(f"  {query}: {len(matches)} matches")
                if matches:
                    best = matches[0]  # The match IS the entity
                    logger.info(f"    Best: {best.get('item_name', 'N/A')} (score: {best.get('score', 0):.3f})")
            
            return {
                "database_results": results,
                "milvus_results": results
            }
            
        except Exception as e:
            logger.error(f"Milvus search failed: {e}", exc_info=True)
            return {
                "database_results": [],
                "milvus_results": [],
                "error": f"Database search failed: {e}",
                "error_stage": "database_lookup"
            }
        
    except Exception as e:
        logger.error(f"❌ Database lookup failed: {e}", exc_info=True)
        return {
            "error": str(e),
            "error_stage": "database_lookup"
        }


# ============================================================================
# NODE 3: LLM Analysis with PROPER CALORIE CALCULATION
# ============================================================================

def llm_analysis_node(state: NutritionWorkflowState) -> Dict[str, Any]:
    """
    Analyze nutrition using LLM with PROPER CALORIE CALCULATION.
    
    Key fix: Multiply portion size (grams) by nutritional values per 100g,
    then sum all ingredients to get total calories.
    """
    try:
        logger.info("🤖 Starting LLM analysis with calorie calculation...")
        
        portions = state.get("portions", {})
        database_results = state.get("database_results", [])
        parsed_json = state.get("parsed_json", {})
        
        if not portions:
            logger.warning("No portions to analyze")
            return {"llm_analysis": None}
        
        if not database_results:
            logger.warning("No database results to calculate nutrition")
            return {"llm_analysis": None}
        
        # =================================================================
        # CALCULATE NUTRITION FOR EACH INGREDIENT
        # =================================================================
        logger.info("📊 Calculating nutrition for each ingredient...")
        
        ingredient_nutrition = []
        total_calories = 0.0
        total_protein = 0.0
        total_carbs = 0.0
        total_fat = 0.0
        total_fiber = 0.0
        
        for db_result in database_results:
            query = db_result.get("query", "")
            matches = db_result.get("results", [])
            
            if not matches:
                logger.warning(f"No matches for: {query}")
                continue
            
            # Get best match (highest score)
            best_match = matches[0]
            entity = best_match  # The match object IS the entity
            score = entity.get("score", 0.0)
            
            # Get nutrition data (per 100g from database)
            # NOTE: Database uses specific field names
            food_name = entity.get("item_name", entity.get("name", query))
            calories_per_100g = entity.get("calories", 0.0)
            protein_per_100g = entity.get("protein_g", 0.0)
            carbs_per_100g = entity.get("carb_g", 0.0)
            fat_per_100g = entity.get("fat_g", 0.0)
            fiber_per_100g = entity.get("fiber_g", 0.0)
            
            # Find corresponding portion (in grams)
            # Query might be cleaned, so try both versions
            portion_grams = 0.0
            
            # Try exact match first
            if query in portions:
                portion_grams = portions[query]
            else:
                # Try to find by matching cleaned names
                query_cleaned = query.replace(' ', '_').replace('-', '_')
                for ing_name, ing_grams in portions.items():
                    ing_cleaned = ing_name.replace(' ', '_').replace('-', '_')
                    if query_cleaned == ing_cleaned or query in ing_name or ing_name in query:
                        portion_grams = ing_grams
                        break
            
            if portion_grams == 0:
                logger.warning(f"Could not find portion for '{query}' in portions: {list(portions.keys())}")
                continue
            
            # CALCULATE actual nutrition based on portion size
            # Formula: (nutrition_per_100g * portion_grams) / 100
            multiplier = portion_grams / 100.0
            
            actual_calories = calories_per_100g * multiplier
            actual_protein = protein_per_100g * multiplier
            actual_carbs = carbs_per_100g * multiplier
            actual_fat = fat_per_100g * multiplier
            actual_fiber = fiber_per_100g * multiplier
            
            logger.info(f"  {query} ({portion_grams}g):")
            logger.info(f"    DB: {calories_per_100g} cal/100g → Actual: {actual_calories:.1f} cal")
            logger.info(f"    Protein: {actual_protein:.1f}g | Carbs: {actual_carbs:.1f}g | Fat: {actual_fat:.1f}g")
            
            # Add to ingredient list
            ingredient_nutrition.append({
                "name": food_name,
                "portion_grams": portion_grams,
                "match_score": score,
                "calories": actual_calories,
                "protein": actual_protein,
                "carbohydrates": actual_carbs,
                "fat": actual_fat,
                "fiber": actual_fiber,
                # Include per-100g values for reference
                "per_100g": {
                    "calories": calories_per_100g,
                    "protein": protein_per_100g,
                    "carbohydrates": carbs_per_100g,
                    "fat": fat_per_100g,
                    "fiber": fiber_per_100g
                }
            })
            
            # Sum totals
            total_calories += actual_calories
            total_protein += actual_protein
            total_carbs += actual_carbs
            total_fat += actual_fat
            total_fiber += actual_fiber
        
        # =================================================================
        # BUILD FINAL ANALYSIS
        # =================================================================
        logger.info(f"✅ Calculated totals:")
        logger.info(f"  Total calories: {total_calories:.1f}")
        logger.info(f"  Total protein: {total_protein:.1f}g")
        logger.info(f"  Total carbs: {total_carbs:.1f}g")
        logger.info(f"  Total fat: {total_fat:.1f}g")
        logger.info(f"  Total fiber: {total_fiber:.1f}g")
        
        llm_analysis = {
            "dish_name": parsed_json.get("dish_name", "Unknown Dish"),
            "food_type": parsed_json.get("food_type", "unknown"),
            "cooking_method": parsed_json.get("cooking_method", "unknown"),
            "ingredients": ingredient_nutrition,
            "total_nutrition": {
                "calories": round(total_calories, 1),
                "protein": round(total_protein, 1),
                "carbohydrates": round(total_carbs, 1),
                "fat": round(total_fat, 1),
                "fiber": round(total_fiber, 1)
            },
            "metadata": {
                "total_ingredients": len(ingredient_nutrition),
                "camera_prob": parsed_json.get("camera_or_phone_prob", 0.0),
                "food_prob": parsed_json.get("food_prob", 0.0)
            }
        }
        
        logger.info("✅ LLM analysis completed with proper calorie calculation")
        
        # =================================================================
        # Generate conversational response about the nutrition
        # =================================================================
        logger.info("💬 Generating conversational analysis...")
        
        try:
            import torch
            from app.services.model_adapter import get_model_and_processor
            
            # Build a prompt for conversational analysis
            conversation_prompt = f"""You are a friendly, knowledgeable nutrition consultant and user assistant called Core Care, You are having a conversation with the user about their food choice.

They just ate/are planning to eat: {llm_analysis['dish_name']}

Nutritional Information:
- Total Calories: {llm_analysis['total_nutrition']['calories']:.0f} kcal
- Protein: {llm_analysis['total_nutrition']['protein']:.1f}g
- Carbohydrates: {llm_analysis['total_nutrition']['carbohydrates']:.1f}g
- Fat: {llm_analysis['total_nutrition']['fat']:.1f}g
- Food Type: {llm_analysis['food_type']}
- Cooking Method: {llm_analysis['cooking_method']}

Ingredients breakdown:
"""
            for ing in llm_analysis['ingredients']:
                conversation_prompt += f"- {ing['name']}: {ing['portion_grams']}g ({ing['calories']:.0f} cal)\n"
            instruction = state.get("instruction")

            conversation_prompt += f"""

Write a friendly, conversational response to the user's question:
1. Acknowledges their food choice.
2. Highlights interesting nutritional aspects (calories, macros, cooking method)
3. Gives gentle, personalized advice about including this in their diet
4. Mentions specific ingredients if relevant
5. Be conversational and natural.

   • Calories: {llm_analysis['total_nutrition']['calories']:.1f} kcal
   • Protein: {llm_analysis['total_nutrition']['protein']:.1f}g
   • Carbs: {llm_analysis['total_nutrition']['carbohydrates']:.1f}g
   • Fat: {llm_analysis['total_nutrition']['fat']:.1f}g"


IMPORTANT: The nutritional list should be naturally woven into the conversation, not in a separate section. Use bullet points (•) for the nutrition list to make it scannable.

Be conversational and natural. Write like you're chatting with a friend who wants to know what they're eating.
USER QUESION:
 {instruction} 
 """

            # Get text-only model for generation
            result = get_model_and_processor(has_image=False)
            if isinstance(result, tuple):
                model, processor = result
            elif isinstance(result, dict):
                model = result.get('model')
                processor = result.get('processor')
            else:
                raise ValueError(f"Unexpected return type: {type(result)}")
            
            # Create messages
            messages = [
                {
                    "role": "system",
                    "content": "You are a friendly, supportive nutritionist called core care who provides helpful dietary advice in a conversational, non-judgmental way."
                },
                {
                    "role": "user",
                    "content": conversation_prompt
                }
            ]
            
            # Apply chat template
            text_prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Get device
            if hasattr(model, 'device'):
                device = model.device
            elif hasattr(model, 'model') and hasattr(model.model, 'device'):
                device = model.model.device
            else:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Process and generate
            inputs = processor(
                text=[text_prompt],
                return_tensors="pt",
                padding=True
            ).to(device)
            
            logger.info("🚀 Generating conversational response...")
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,  # Slightly higher for more natural conversation
                    top_p=0.9
                )
            
            # Decode
            tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
            generated_text = tokenizer.batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )[0]
            
            # Extract response
            conversational_response = generated_text.split("assistant\n")[-1].strip()
            
            # Cleanup
            del inputs, output_ids
            if device.type == "cuda":
                torch.cuda.empty_cache()
            
            logger.info(f"✅ Generated {len(conversational_response)} characters of conversation")
            logger.info(f"Preview: {conversational_response[:200]}...")
            
            # Add to analysis
            llm_analysis["conversation"] = conversational_response
            
        except Exception as e:
            logger.error(f"Failed to generate conversation: {e}", exc_info=True)
            # Provide a fallback message with nutrition breakdown
            llm_analysis["conversation"] = (
                f"Great choice! Your {llm_analysis['dish_name']} looks delicious. "
                f"Here's the nutritional breakdown:\n\n"
                f"• Calories: {llm_analysis['total_nutrition']['calories']:.1f} kcal\n"
                f"• Protein: {llm_analysis['total_nutrition']['protein']:.1f}g\n"
                f"• Carbs: {llm_analysis['total_nutrition']['carbohydrates']:.1f}g\n"
                f"• Fat: {llm_analysis['total_nutrition']['fat']:.1f}g\n\n"
                f"This {llm_analysis['food_type']} dish can definitely be part of a balanced diet. "
                f"Enjoy your meal!"
            )
        
        logger.info("✅ Complete analysis with conversation ready")
        
        return {
            "llm_analysis": llm_analysis,
            "nutrition_result": llm_analysis
        }
        
    except Exception as e:
        logger.error(f"❌ LLM analysis failed: {e}", exc_info=True)
        return {
            "error": str(e),
            "error_stage": "llm_analysis"
        }


# ============================================================================
# WORKFLOW BUILDER
# ============================================================================

def build_nutrition_workflow():
    """
    Build the complete nutrition analysis workflow.
    Supports both image and text-only inputs.
    """
    logger.info("Building nutrition workflow with fixed calorie calculation...")
    
    workflow = StateGraph(NutritionWorkflowState)
    
    # Add nodes
    workflow.add_node("vision_analysis", vision_analysis_node)
    workflow.add_node("database_lookup", database_lookup_node)
    workflow.add_node("llm_analysis", llm_analysis_node)
    
    # Define edges
    workflow.add_edge("vision_analysis", "database_lookup")
    workflow.add_edge("database_lookup", "llm_analysis")
    workflow.add_edge("llm_analysis", END)
    
    # Set entry point
    workflow.set_entry_point("vision_analysis")
    
    logger.info("✅ Workflow structure: vision_analysis → database_lookup → llm_analysis → END")
    
    return workflow.compile()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def validate_workflow_state(state: NutritionWorkflowState) -> bool:
    """Check if workflow state is valid."""
    if state.get("error"):
        logger.error(f"Workflow error at {state.get('error_stage', 'unknown')}: {state['error']}")
        return False
    return True


def get_workflow_summary(state: NutritionWorkflowState) -> str:
    """Get a detailed summary of the workflow state."""
    lines = []
    lines.append("="*60)
    lines.append("WORKFLOW STATE SUMMARY")
    lines.append("="*60)
    
    # Input
    has_image = "✅" if state.get("image") else "❌"
    lines.append(f"Image: {has_image}")
    lines.append(f"Instruction: {state.get('instruction', 'None')[:50]}...")
    
    # Vision analysis
    has_raw = "✅" if state.get("raw_text") else "❌"
    has_parsed = "✅" if state.get("parsed_json") else "❌"
    portions = state.get("portions", {})
    lines.append(f"Raw text: {has_raw} | Parsed JSON: {has_parsed}")
    lines.append(f"Portions: {len(portions)} ingredients (all in grams)")
    
    if portions:
        lines.append("  Ingredient portions:")
        for name, grams in portions.items():
            lines.append(f"    - {name}: {grams}g")
    
    # Database lookup
    db_results = state.get("milvus_results", [])
    lines.append(f"Database results: {len(db_results)} matches")
    
    # LLM analysis
    llm_analysis = state.get("llm_analysis")
    if llm_analysis:
        lines.append(f"LLM Analysis: ✅")
        lines.append(f"  Dish: {llm_analysis.get('dish_name', 'Unknown')}")
        
        total_nutrition = llm_analysis.get('total_nutrition', {})
        calories = total_nutrition.get('calories', 0)
        protein = total_nutrition.get('protein', 0)
        carbs = total_nutrition.get('carbohydrates', 0)
        fat = total_nutrition.get('fat', 0)
        
        lines.append(f"  Total Calories: {calories:.1f} kcal")
        lines.append(f"  Protein: {protein:.1f}g | Carbs: {carbs:.1f}g | Fat: {fat:.1f}g")
        
        # Show ingredient breakdown
        ingredients = llm_analysis.get('ingredients', [])
        if ingredients:
            lines.append("  Ingredient breakdown:")
            for ing in ingredients:
                name = ing.get('name', 'Unknown')
                portion = ing.get('portion_grams', 0)
                cals = ing.get('calories', 0)
                lines.append(f"    - {name} ({portion}g): {cals:.1f} cal")
        
        # Show conversational response
        conversation = llm_analysis.get('conversation')
        if conversation:
            lines.append("")
            lines.append("💬 Nutritionist's Response:")
            lines.append("─" * 60)
            # Wrap conversation text at 60 characters
            conv_words = conversation.split()
            current_line = "  "
            for word in conv_words:
                if len(current_line) + len(word) + 1 <= 62:
                    current_line += word + " "
                else:
                    lines.append(current_line.rstrip())
                    current_line = "  " + word + " "
            if current_line.strip():
                lines.append(current_line.rstrip())
            lines.append("─" * 60)
    else:
        lines.append(f"LLM Analysis: ❌")
    
    # Errors
    if state.get("error"):
        lines.append(f"❌ ERROR: {state['error']}")
        lines.append(f"   Stage: {state.get('error_stage', 'unknown')}")
    
    lines.append("="*60)
    return "\n".join(lines)



# """
# LangGraph Workflows - FULLY FIXED Version
# ==========================================
# FIXES:
# 1. ✅ All portions converted to grams (ml → g, oz → g, etc.)
# 2. ✅ Total calories properly calculated from portions × database nutrition data
# 3. ✅ Supports both image and text-only inputs
# """
# from typing import TypedDict, Dict, Any, Optional, List
# from langgraph.graph import StateGraph, END
# from PIL import Image
# import io
# import logging
# import re

# logger = logging.getLogger(__name__)


# # ============================================================================
# # STATE SCHEMA
# # ============================================================================

# class NutritionWorkflowState(TypedDict, total=False):
#     """State for the nutrition analysis workflow."""
#     # Input fields
#     image: Optional[bytes]
#     instruction: str
#     system_prompt: Optional[str]
    
#     # Vision analysis output
#     raw_text: str
#     parsed_json: Optional[Dict[str, Any]]
#     portions: Optional[Dict[str, float]]  # ALL IN GRAMS
    
#     # Database lookup output
#     database_results: Optional[List[Dict[str, Any]]]
#     milvus_results: Optional[List[Dict[str, Any]]]
    
#     # LLM analysis output
#     llm_analysis: Optional[Dict[str, Any]]
    
#     # Final nutrition result
#     nutrition_result: Optional[Dict[str, Any]]
    
#     # Error handling
#     error: Optional[str]
#     error_stage: Optional[str]


# # ============================================================================
# # UNIT CONVERSION UTILITIES
# # ============================================================================

# def convert_to_grams(amount: float, unit: str) -> float:
#     """
#     Convert various units to grams.
    
#     Args:
#         amount: The numeric amount
#         unit: The unit (g, ml, oz, lb, cup, tbsp, tsp, etc.)
    
#     Returns:
#         Amount in grams
#     """
#     unit = unit.lower().strip()
    
#     # Already in grams
#     if unit in ['g', 'gram', 'grams', 'gr']:
#         return amount
    
#     # Milliliters (assume density of water: 1ml = 1g)
#     # For oils: 1ml ≈ 0.92g, but we'll use 1:1 for simplicity
#     if unit in ['ml', 'milliliter', 'milliliters', 'mL']:
#         return amount
    
#     # Liters
#     if unit in ['l', 'liter', 'liters', 'L']:
#         return amount * 1000
    
#     # Ounces (weight)
#     if unit in ['oz', 'ounce', 'ounces']:
#         return amount * 28.35
    
#     # Pounds
#     if unit in ['lb', 'lbs', 'pound', 'pounds']:
#         return amount * 453.592
    
#     # Cups (approximate, varies by ingredient)
#     if unit in ['cup', 'cups', 'c']:
#         return amount * 240  # Assume liquid cup
    
#     # Tablespoons
#     if unit in ['tbsp', 'tablespoon', 'tablespoons', 'T']:
#         return amount * 15
    
#     # Teaspoons
#     if unit in ['tsp', 'teaspoon', 'teaspoons', 't']:
#         return amount * 5
    
#     # Kilograms
#     if unit in ['kg', 'kilogram', 'kilograms']:
#         return amount * 1000
    
#     # If unknown unit, log warning and return original amount
#     logger.warning(f"Unknown unit '{unit}', treating as grams")
#     return amount


# def parse_portion_string(portion_str: str) -> tuple[str, float]:
#     """
#     Parse a portion string like "chicken_breast:250g" or "olive_oil:15ml"
#     and return (ingredient_name, grams).
    
#     Args:
#         portion_str: String in format "ingredient:amount_unit"
    
#     Returns:
#         Tuple of (ingredient_name, amount_in_grams)
#     """
#     try:
#         if ':' not in portion_str:
#             logger.warning(f"Invalid portion format (no colon): {portion_str}")
#             return None, 0.0
        
#         # Split into ingredient and amount
#         ingredient, amount_str = portion_str.split(':', 1)
#         ingredient = ingredient.strip()
#         amount_str = amount_str.strip()
        
#         # Extract number and unit using regex
#         # Matches patterns like: "250g", "15ml", "2.5oz", "1.5 cup"
#         match = re.match(r'(\d+(?:\.\d+)?)\s*([a-zA-Z]+)?', amount_str)
        
#         if not match:
#             logger.warning(f"Could not parse amount from: {amount_str}")
#             return ingredient, 100.0  # Default 100g
        
#         amount = float(match.group(1))
#         unit = match.group(2) if match.group(2) else 'g'  # Default to grams if no unit
        
#         # Convert to grams
#         grams = convert_to_grams(amount, unit)
        
#         logger.info(f"  Parsed: {ingredient} = {amount}{unit} → {grams}g")
#         return ingredient, grams
        
#     except Exception as e:
#         logger.error(f"Error parsing portion '{portion_str}': {e}")
#         return None, 0.0


# # ============================================================================
# # NODE 1: Vision Analysis (supports both image and text-only)
# # ============================================================================

# def vision_analysis_node(state: NutritionWorkflowState) -> Dict[str, Any]:
#     """
#     Analyze food using vision model (if image provided) or process text-only query.
#     ALL PORTIONS RETURNED IN GRAMS.
#     """
#     import torch
#     import json
    
#     try:
#         logger.info("🔍 Starting vision analysis...")
        
#         # Get input data
#         image_bytes = state.get("image")
#         instruction = state.get("instruction", "Analyze the ingredients in this food.")
#         system_prompt = state.get("system_prompt")
        
#         # Force JSON output if not explicitly requesting it
#         if "json" not in instruction.lower() and "JSON" not in instruction:
#             if image_bytes:
#                 json_instruction = f"""{instruction}

# CRITICAL: Respond with ONLY a valid JSON object. No other text, explanations, or markdown. Start directly with {{ and end with }}.

# Use this EXACT format (ALL AMOUNTS MUST INCLUDE UNITS):
# {{
#   "camera_or_phone_prob": <float 0.00-1.00>,
#   "food_prob": <float 0.00-1.00>,
#   "dish_name": "<lowercase camel case string>",
#   "food_type": "<one of: home_cooked | restaurant_food | packaged | dessert | beverage | baked_goods>",
#   "ingredients": ["<lowercase_snake_case strings>"],
#   "portion_size": ["<item>:<amount><unit>"],
#   "cooking_method": "<one of: frying | baking | grilling | boiling | steaming | roasting | raw | sauteing | stew>"
# }}

# IMPORTANT: In portion_size, ALWAYS include the unit in gram (g).

# Example:
# {{
#   "camera_or_phone_prob": 0.95,
#   "food_prob": 0.98,
#   "dish_name": "grilledChickenKebab",
#   "food_type": "restaurant_food",
#   "ingredients": ["chicken_breast", "olive_oil", "lemon", "onion"],
#   "portion_size": ["chicken_breast:250g", "olive_oil:15ml", "lemon:20g", "onion:50g"],
#   "cooking_method": "grilling"
# }}"""
#                 instruction = json_instruction
#                 logger.info("✅ Added JSON formatting requirement to instruction")
        
#         # ===================================================================
#         # CASE 1: No image - text-only query
#         # ===================================================================
#         if not image_bytes:
#             logger.info("📝 No image provided - text-only mode")
#             logger.info(f"Processing instruction: {instruction}")
            
#             from app.services.model_adapter import get_model_and_processor
#             result = get_model_and_processor(has_image=False)
            
#             if isinstance(result, tuple):
#                 model, processor = result
#                 logger.info("✅ Unpacked model and processor from tuple")
#             elif isinstance(result, dict):
#                 model = result.get('model')
#                 processor = result.get('processor')
#                 logger.info("✅ Extracted model and processor from dict")
#             else:
#                 raise ValueError(f"Unexpected return type from get_model_and_processor: {type(result)}")
            
#             logger.info(f"✅ Model loaded for text-only: {type(model).__name__}")
            
#             text_prompt_content = f"""Analyze this food description and extract ingredients with their estimated portions.

# Food description: {instruction}

# CRITICAL: You MUST respond with ONLY a valid JSON object. Do NOT include any other text, explanations, or markdown.

# Use this EXACT format (INCLUDE UNITS):
# {{
#   "camera_or_phone_prob": <float>,
#   "food_prob": <float>,
#   "dish_name": "<lowercase camel case>",
#   "food_type": "<type>",
#   "ingredients": ["<ingredients>"],
#   "portion_size": ["<item>:<amount><UNIT>"],
#   "cooking_method": "<method>"
# }}

# IMPORTANT: Always include units in portion_size in gram only (g)"""
            
#             messages = [
#                 {
#                     "role": "system",
#                     "content": system_prompt or "You are a nutrition expert. Extract ingredients and portions from food descriptions."
#                 },
#                 {
#                     "role": "user",
#                     "content": text_prompt_content
#                 }
#             ]
            
#             logger.info("🤖 Applying chat template for text-only analysis...")
#             text_prompt = processor.apply_chat_template(
#                 messages,
#                 tokenize=False,
#                 add_generation_prompt=True
#             )
            
#             # Get device
#             if hasattr(model, 'device'):
#                 device = model.device
#             elif hasattr(model, 'model') and hasattr(model.model, 'device'):
#                 device = model.model.device
#             else:
#                 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
#             inputs = processor(
#                 text=[text_prompt],
#                 return_tensors="pt",
#                 padding=True
#             ).to(device)
            
#             logger.info("🚀 Generating text-only analysis...")
#             with torch.no_grad():
#                 output_ids = model.generate(
#                     **inputs,
#                     max_new_tokens=512,
#                     do_sample=True,
#                     temperature=0.5,
#                     top_p=0.9
#                 )
            
#             tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
#             generated_text = tokenizer.batch_decode(
#                 output_ids,
#                 skip_special_tokens=True,
#                 clean_up_tokenization_spaces=True
#             )[0]
            
#             response_text = generated_text.split("assistant\n")[-1].strip()
            
#             del inputs, output_ids
#             if device.type == "cuda":
#                 torch.cuda.empty_cache()
        
#         # ===================================================================
#         # CASE 2: Image provided - vision analysis
#         # ===================================================================
#         else:
#             logger.info("📦 Converting bytes to PIL Image...")
#             image = Image.open(io.BytesIO(image_bytes))
#             if image.mode != "RGB":
#                 image = image.convert("RGB")
#             logger.info(f"✅ Image converted: {image.size}, mode: {image.mode}")
            
#             from app.services.model_adapter import get_model_and_processor
#             result = get_model_and_processor(has_image=True)
            
#             if isinstance(result, tuple):
#                 model, processor = result
#                 logger.info("✅ Unpacked model and processor from tuple")
#             elif isinstance(result, dict):
#                 model = result.get('model')
#                 processor = result.get('processor')
#                 logger.info("✅ Extracted model and processor from dict")
#             else:
#                 raise ValueError(f"Unexpected return type: {type(result)}")
            
#             logger.info(f"✅ Model loaded: {type(model).__name__}")
            
#             conversation = []
#             if system_prompt:
#                 conversation.append({"role": "system", "content": system_prompt})
            
#             conversation.append({
#                 "role": "user",
#                 "content": [
#                     {"type": "image", "image": image},
#                     {"type": "text", "text": instruction}
#                 ]
#             })
            
#             logger.info("🤖 Applying chat template...")
#             text_prompt = processor.apply_chat_template(
#                 conversation,
#                 tokenize=False,
#                 add_generation_prompt=True
#             )
            
#             if hasattr(model, 'device'):
#                 device = model.device
#             elif hasattr(model, 'model') and hasattr(model.model, 'device'):
#                 device = model.model.device
#             else:
#                 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
#             logger.info("📝 Processing inputs...")
#             inputs = processor(
#                 text=[text_prompt],
#                 images=[image],
#                 return_tensors="pt",
#                 padding=True
#             ).to(device)
            
#             logger.info("🚀 Generating vision analysis...")
#             with torch.no_grad():
#                 output_ids = model.generate(
#                     **inputs,
#                     max_new_tokens=1024,
#                     do_sample=True,
#                     temperature=0.5,
#                     top_p=0.9
#                 )
            
#             logger.info("📤 Decoding output...")
#             generated_text = processor.batch_decode(
#                 output_ids,
#                 skip_special_tokens=True,
#                 clean_up_tokenization_spaces=True
#             )[0]
            
#             response_text = generated_text.split("assistant\n")[-1].strip()
            
#             del inputs, output_ids
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()
        
#         # ===================================================================
#         # Common processing - Parse JSON and extract portions IN GRAMS
#         # ===================================================================
        
#         logger.info(f"✅ Generated {len(response_text)} characters")
#         logger.info(f"Raw response preview: {response_text[:300]}...")
        
#         # Parse JSON from response
#         json_str = None
#         parsed_json = None
        
#         # Try multiple extraction methods
#         json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
#         if json_match:
#             json_str = json_match.group(1)
#             logger.info("✅ Found JSON in markdown code block")
        
#         if not json_str:
#             json_match = re.search(r'```\s*(\{.*?\})\s*```', response_text, re.DOTALL)
#             if json_match:
#                 json_str = json_match.group(1)
#                 logger.info("✅ Found JSON in plain code block")
        
#         if not json_str:
#             start = response_text.find('{')
#             end = response_text.rfind('}')
#             if start != -1 and end != -1 and start < end:
#                 json_str = response_text[start:end+1]
#                 logger.info("✅ Extracted JSON from { to }")
        
#         if not json_str:
#             logger.warning("⚠️ No JSON delimiters found, trying entire response")
#             json_str = response_text
        
#         try:
#             parsed_json = json.loads(json_str)
#             logger.info("✅ JSON parsed successfully")
#         except json.JSONDecodeError as e:
#             logger.error(f"❌ JSON parsing failed: {e}")
#             logger.info("🔧 Attempting JSON cleanup...")
#             try:
#                 cleaned = json_str[json_str.find('{'):]
#                 cleaned = cleaned[:cleaned.rfind('}')+1]
#                 cleaned = cleaned.replace('**', '')
#                 parsed_json = json.loads(cleaned)
#                 logger.info("✅ JSON parsed after cleanup")
#             except:
#                 logger.error("❌ JSON cleanup failed")
#                 return {
#                     "error": f"Failed to parse JSON: {e}",
#                     "error_stage": "vision_analysis",
#                     "raw_text": response_text
#                 }
        
#         # =================================================================
#         # Extract portions and CONVERT ALL TO GRAMS
#         # =================================================================
#         logger.info("📊 Parsing portions and converting to grams...")
#         portions = {}
        
#         portion_size_list = parsed_json.get("portion_size", [])
#         if portion_size_list:
#             for portion_str in portion_size_list:
#                 ingredient, grams = parse_portion_string(portion_str)
#                 if ingredient and grams > 0:
#                     portions[ingredient] = grams
        
#         # Fallback: use ingredients with default 100g
#         if not portions:
#             logger.warning("⚠️ No portion_size field, using default 100g per ingredient")
#             ingredients_list = parsed_json.get("ingredients", [])
#             for ing in ingredients_list:
#                 portions[ing] = 100.0
#                 logger.info(f"  - {ing}: 100g (default)")
        
#         logger.info(f"✅ Extracted {len(portions)} ingredients (ALL IN GRAMS):")
#         for name, grams in portions.items():
#             logger.info(f"  - {name}: {grams}g")
        
#         # Log additional fields
#         camera_prob = parsed_json.get("camera_or_phone_prob", 0.0)
#         food_prob = parsed_json.get("food_prob", 0.0)
#         food_type = parsed_json.get("food_type", "unknown")
#         cooking_method = parsed_json.get("cooking_method", "unknown")
        
#         logger.info(f"📸 Camera probability: {camera_prob:.2f}")
#         logger.info(f"🍽️  Food probability: {food_prob:.2f}")
#         logger.info(f"📋 Food type: {food_type}")
#         logger.info(f"🔥 Cooking method: {cooking_method}")
        
#         # Add portions to parsed_json
#         parsed_json["portions"] = portions
        
#         return {
#             "raw_text": response_text,
#             "parsed_json": parsed_json,
#             "portions": portions  # ALL IN GRAMS
#         }
        
#     except Exception as e:
#         logger.error(f"❌ Vision analysis failed: {e}", exc_info=True)
#         return {
#             "error": str(e),
#             "error_stage": "vision_analysis"
#         }


# # ============================================================================
# # NODE 2: Database Lookup
# # ============================================================================

# def database_lookup_node(state: NutritionWorkflowState) -> Dict[str, Any]:
#     """Search database for top-3 matches for each ingredient."""
#     from app.services.milvus_service import MilvusManager
    
#     try:
#         logger.info("🔍 Starting database lookup...")
        
#         portions = state.get("portions", {})
        
#         if not portions:
#             logger.warning("No portions found, skipping database lookup")
#             return {
#                 "database_results": [],
#                 "milvus_results": []
#             }
        
#         ingredient_names = list(portions.keys())
#         logger.info(f"Looking up {len(ingredient_names)} ingredients: {ingredient_names}")
        
#         # Clean ingredient names
#         cleaned_ingredient_names = [
#             name.replace('_', ' ').replace('-', ' ') 
#             for name in ingredient_names
#         ]
#         logger.info(f"Cleaned names for search: {cleaned_ingredient_names}")
        
#         milvus = MilvusManager.get_instance()
        
#         try:
#             results = milvus.search_ingredients(cleaned_ingredient_names, top_k=3)
#             logger.info(f"✅ Found {len(results)} ingredient matches")
            
#             for result in results:
#                 query = result.get("query", "Unknown")
#                 matches = result.get("results", [])
#                 logger.info(f"  {query}: {len(matches)} matches")
#                 if matches:
#                     best = matches[0]  # The match IS the entity
#                     logger.info(f"    Best: {best.get('item_name', 'N/A')} (score: {best.get('score', 0):.3f})")
            
#             return {
#                 "database_results": results,
#                 "milvus_results": results
#             }
            
#         except Exception as e:
#             logger.error(f"Milvus search failed: {e}", exc_info=True)
#             return {
#                 "database_results": [],
#                 "milvus_results": [],
#                 "error": f"Database search failed: {e}",
#                 "error_stage": "database_lookup"
#             }
        
#     except Exception as e:
#         logger.error(f"❌ Database lookup failed: {e}", exc_info=True)
#         return {
#             "error": str(e),
#             "error_stage": "database_lookup"
#         }


# # ============================================================================
# # NODE 3: LLM Analysis with PROPER CALORIE CALCULATION
# # ============================================================================

# def llm_analysis_node(state: NutritionWorkflowState) -> Dict[str, Any]:
#     """
#     Analyze nutrition using LLM with PROPER CALORIE CALCULATION.
    
#     Key fix: Multiply portion size (grams) by nutritional values per 100g,
#     then sum all ingredients to get total calories.
#     """
#     try:
#         logger.info("🤖 Starting LLM analysis with calorie calculation...")
        
#         portions = state.get("portions", {})
#         database_results = state.get("database_results", [])
#         parsed_json = state.get("parsed_json", {})
        
#         if not portions:
#             logger.warning("No portions to analyze")
#             return {"llm_analysis": None}
        
#         if not database_results:
#             logger.warning("No database results to calculate nutrition")
#             return {"llm_analysis": None}
        
#         # =================================================================
#         # CALCULATE NUTRITION FOR EACH INGREDIENT
#         # =================================================================
#         logger.info("📊 Calculating nutrition for each ingredient...")
        
#         ingredient_nutrition = []
#         total_calories = 0.0
#         total_protein = 0.0
#         total_carbs = 0.0
#         total_fat = 0.0
#         total_fiber = 0.0
        
#         for db_result in database_results:
#             query = db_result.get("query", "")
#             matches = db_result.get("results", [])
            
#             if not matches:
#                 logger.warning(f"No matches for: {query}")
#                 continue
            
#             # Get best match (highest score)
#             best_match = matches[0]
#             entity = best_match  # The match object IS the entity
#             score = entity.get("score", 0.0)
            
#             # Get nutrition data (per 100g from database)
#             # NOTE: Database uses specific field names
#             food_name = entity.get("item_name", entity.get("name", query))
#             calories_per_100g = entity.get("calories", 0.0)
#             protein_per_100g = entity.get("protein_g", 0.0)
#             carbs_per_100g = entity.get("carb_g", 0.0)
#             fat_per_100g = entity.get("fat_g", 0.0)
#             fiber_per_100g = entity.get("fiber_g", 0.0)
            
#             # Find corresponding portion (in grams)
#             # Query might be cleaned, so try both versions
#             portion_grams = 0.0
            
#             # Try exact match first
#             if query in portions:
#                 portion_grams = portions[query]
#             else:
#                 # Try to find by matching cleaned names
#                 query_cleaned = query.replace(' ', '_').replace('-', '_')
#                 for ing_name, ing_grams in portions.items():
#                     ing_cleaned = ing_name.replace(' ', '_').replace('-', '_')
#                     if query_cleaned == ing_cleaned or query in ing_name or ing_name in query:
#                         portion_grams = ing_grams
#                         break
            
#             if portion_grams == 0:
#                 logger.warning(f"Could not find portion for '{query}' in portions: {list(portions.keys())}")
#                 continue
            
#             # CALCULATE actual nutrition based on portion size
#             # Formula: (nutrition_per_100g * portion_grams) / 100
#             multiplier = portion_grams / 100.0
            
#             actual_calories = calories_per_100g * multiplier
#             actual_protein = protein_per_100g * multiplier
#             actual_carbs = carbs_per_100g * multiplier
#             actual_fat = fat_per_100g * multiplier
#             actual_fiber = fiber_per_100g * multiplier
            
#             logger.info(f"  {query} ({portion_grams}g):")
#             logger.info(f"    DB: {calories_per_100g} cal/100g → Actual: {actual_calories:.1f} cal")
#             logger.info(f"    Protein: {actual_protein:.1f}g | Carbs: {actual_carbs:.1f}g | Fat: {actual_fat:.1f}g")
            
#             # Add to ingredient list
#             ingredient_nutrition.append({
#                 "name": food_name,
#                 "portion_grams": portion_grams,
#                 "match_score": score,
#                 "calories": actual_calories,
#                 "protein": actual_protein,
#                 "carbohydrates": actual_carbs,
#                 "fat": actual_fat,
#                 "fiber": actual_fiber,
#                 # Include per-100g values for reference
#                 "per_100g": {
#                     "calories": calories_per_100g,
#                     "protein": protein_per_100g,
#                     "carbohydrates": carbs_per_100g,
#                     "fat": fat_per_100g,
#                     "fiber": fiber_per_100g
#                 }
#             })
            
#             # Sum totals
#             total_calories += actual_calories
#             total_protein += actual_protein
#             total_carbs += actual_carbs
#             total_fat += actual_fat
#             total_fiber += actual_fiber
        
#         # =================================================================
#         # BUILD FINAL ANALYSIS
#         # =================================================================
#         logger.info(f"✅ Calculated totals:")
#         logger.info(f"  Total calories: {total_calories:.1f}")
#         logger.info(f"  Total protein: {total_protein:.1f}g")
#         logger.info(f"  Total carbs: {total_carbs:.1f}g")
#         logger.info(f"  Total fat: {total_fat:.1f}g")
#         logger.info(f"  Total fiber: {total_fiber:.1f}g")
        
#         llm_analysis = {
#             "dish_name": parsed_json.get("dish_name", "Unknown Dish"),
#             "food_type": parsed_json.get("food_type", "unknown"),
#             "cooking_method": parsed_json.get("cooking_method", "unknown"),
#             "ingredients": ingredient_nutrition,
#             "total_nutrition": {
#                 "calories": round(total_calories, 1),
#                 "protein": round(total_protein, 1),
#                 "carbohydrates": round(total_carbs, 1),
#                 "fat": round(total_fat, 1),
#                 "fiber": round(total_fiber, 1)
#             },
#             "metadata": {
#                 "total_ingredients": len(ingredient_nutrition),
#                 "camera_prob": parsed_json.get("camera_or_phone_prob", 0.0),
#                 "food_prob": parsed_json.get("food_prob", 0.0)
#             }
#         }
        
#         logger.info("✅ LLM analysis completed with proper calorie calculation")
        
#         # =================================================================
#         # Generate conversational response about the nutrition
#         # =================================================================
#         logger.info("💬 Generating conversational analysis...")
        
#         try:
#             import torch
#             from app.services.model_adapter import get_model_and_processor
#             instruction = state.get("instruction")
#             # Build a prompt for conversational analysis
#             conversation_prompt = f"""You are a friendly, knowledgeable nutrition consultant and user assistant called Core Care, You are having a conversation with the user about their food choice.

# They just ate/are planning to eat: {llm_analysis['dish_name']}

# Nutritional Information:
# - Total Calories: {llm_analysis['total_nutrition']['calories']:.0f} kcal
# - Protein: {llm_analysis['total_nutrition']['protein']:.1f}g
# - Carbohydrates: {llm_analysis['total_nutrition']['carbohydrates']:.1f}g
# - Fat: {llm_analysis['total_nutrition']['fat']:.1f}g
# - Food Type: {llm_analysis['food_type']}
# - Cooking Method: {llm_analysis['cooking_method']}

# Ingredients breakdown:
# """
#             for ing in llm_analysis['ingredients']:
#                 conversation_prompt += f"- {ing['name']}: {ing['portion_grams']}g ({ing['calories']:.0f} cal)\n"
            
#             conversation_prompt += f"""

# Write a friendly, conversational response to the user's question:
# 1. Acknowledges their food choice.
# 2. Highlights interesting nutritional aspects (calories, macros, cooking method)
# 3. Gives gentle, personalized advice about including this in their diet
# 4. Mentions specific ingredients if relevant
# 5. Be conversational and natural.

# USER QUESION:
#  {instruction} 
# """

#             # Get text-only model for generation
#             result = get_model_and_processor(has_image=False)
#             if isinstance(result, tuple):
#                 model, processor = result
#             elif isinstance(result, dict):
#                 model = result.get('model')
#                 processor = result.get('processor')
#             else:
#                 raise ValueError(f"Unexpected return type: {type(result)}")
            
#             # Create messages
#             messages = [
#                 {
#                     "role": "system",
#                     "content": "You are a friendly, supportive nutritionist who provides helpful dietary advice in a conversational, non-judgmental way."
#                 },
#                 {
#                     "role": "user",
#                     "content": conversation_prompt
#                 }
#             ]
            
#             # Apply chat template
#             text_prompt = processor.apply_chat_template(
#                 messages,
#                 tokenize=False,
#                 add_generation_prompt=True
#             )
            
#             # Get device
#             if hasattr(model, 'device'):
#                 device = model.device
#             elif hasattr(model, 'model') and hasattr(model.model, 'device'):
#                 device = model.model.device
#             else:
#                 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
#             # Process and generate
#             inputs = processor(
#                 text=[text_prompt],
#                 return_tensors="pt",
#                 padding=True
#             ).to(device)
            
#             logger.info("🚀 Generating conversational response...")
#             with torch.no_grad():
#                 output_ids = model.generate(
#                     **inputs,
#                     max_new_tokens=512,
#                     do_sample=True,
#                     temperature=0.7,  # Slightly higher for more natural conversation
#                     top_p=0.9
#                 )
            
#             # Decode
#             tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
#             generated_text = tokenizer.batch_decode(
#                 output_ids,
#                 skip_special_tokens=True,
#                 clean_up_tokenization_spaces=True
#             )[0]
            
#             # Extract response
#             conversational_response = generated_text.split("assistant\n")[-1].strip()
            
#             # Cleanup
#             del inputs, output_ids
#             if device.type == "cuda":
#                 torch.cuda.empty_cache()
            
#             logger.info(f"✅ Generated {len(conversational_response)} characters of conversation")
#             logger.info(f"Preview: {conversational_response[:200]}...")
            
#             # Add to analysis
#             llm_analysis["conversation"] = conversational_response
            
#         except Exception as e:
#             logger.error(f"Failed to generate conversation: {e}", exc_info=True)
#             # Provide a fallback simple message
#             llm_analysis["conversation"] = (
#                 f"Great choice! Your {llm_analysis['dish_name']} comes in at "
#                 f"{llm_analysis['total_nutrition']['calories']:.0f} calories with "
#                 f"{llm_analysis['total_nutrition']['protein']:.1f}g of protein. "
#                 f"This {llm_analysis['food_type']} dish can definitely be part of a balanced diet. "
#                 f"Enjoy your meal!"
#             )
        
#         logger.info("✅ Complete analysis with conversation ready")
        
#         return {
#             "llm_analysis": llm_analysis,
#             "nutrition_result": llm_analysis
#         }
        
#     except Exception as e:
#         logger.error(f"❌ LLM analysis failed: {e}", exc_info=True)
#         return {
#             "error": str(e),
#             "error_stage": "llm_analysis"
#         }


# # ============================================================================
# # WORKFLOW BUILDER
# # ============================================================================

# def build_nutrition_workflow():
#     """
#     Build the complete nutrition analysis workflow.
#     Supports both image and text-only inputs.
#     """
#     logger.info("Building nutrition workflow with fixed calorie calculation...")
    
#     workflow = StateGraph(NutritionWorkflowState)
    
#     # Add nodes
#     workflow.add_node("vision_analysis", vision_analysis_node)
#     workflow.add_node("database_lookup", database_lookup_node)
#     workflow.add_node("llm_analysis", llm_analysis_node)
    
#     # Define edges
#     workflow.add_edge("vision_analysis", "database_lookup")
#     workflow.add_edge("database_lookup", "llm_analysis")
#     workflow.add_edge("llm_analysis", END)
    
#     # Set entry point
#     workflow.set_entry_point("vision_analysis")
    
#     logger.info("✅ Workflow structure: vision_analysis → database_lookup → llm_analysis → END")
    
#     return workflow.compile()


# # ============================================================================
# # UTILITY FUNCTIONS
# # ============================================================================

# def validate_workflow_state(state: NutritionWorkflowState) -> bool:
#     """Check if workflow state is valid."""
#     if state.get("error"):
#         logger.error(f"Workflow error at {state.get('error_stage', 'unknown')}: {state['error']}")
#         return False
#     return True


# def get_workflow_summary(state: NutritionWorkflowState) -> str:
#     """Get a detailed summary of the workflow state."""
#     lines = []
#     lines.append("="*60)
#     lines.append("WORKFLOW STATE SUMMARY")
#     lines.append("="*60)
    
#     # Input
#     has_image = "✅" if state.get("image") else "❌"
#     lines.append(f"Image: {has_image}")
#     lines.append(f"Instruction: {state.get('instruction', 'None')[:50]}...")
    
#     # Vision analysis
#     has_raw = "✅" if state.get("raw_text") else "❌"
#     has_parsed = "✅" if state.get("parsed_json") else "❌"
#     portions = state.get("portions", {})
#     lines.append(f"Raw text: {has_raw} | Parsed JSON: {has_parsed}")
#     lines.append(f"Portions: {len(portions)} ingredients (all in grams)")
    
#     if portions:
#         lines.append("  Ingredient portions:")
#         for name, grams in portions.items():
#             lines.append(f"    - {name}: {grams}g")
    
#     # Database lookup
#     db_results = state.get("milvus_results", [])
#     lines.append(f"Database results: {len(db_results)} matches")
    
#     # LLM analysis
#     llm_analysis = state.get("llm_analysis")
#     if llm_analysis:
#         lines.append(f"LLM Analysis: ✅")
#         lines.append(f"  Dish: {llm_analysis.get('dish_name', 'Unknown')}")
        
#         total_nutrition = llm_analysis.get('total_nutrition', {})
#         calories = total_nutrition.get('calories', 0)
#         protein = total_nutrition.get('protein', 0)
#         carbs = total_nutrition.get('carbohydrates', 0)
#         fat = total_nutrition.get('fat', 0)
        
#         lines.append(f"  Total Calories: {calories:.1f} kcal")
#         lines.append(f"  Protein: {protein:.1f}g | Carbs: {carbs:.1f}g | Fat: {fat:.1f}g")
        
#         # Show ingredient breakdown
#         ingredients = llm_analysis.get('ingredients', [])
#         if ingredients:
#             lines.append("  Ingredient breakdown:")
#             for ing in ingredients:
#                 name = ing.get('name', 'Unknown')
#                 portion = ing.get('portion_grams', 0)
#                 cals = ing.get('calories', 0)
#                 lines.append(f"    - {name} ({portion}g): {cals:.1f} cal")
        
#         # Show conversational response
#         conversation = llm_analysis.get('conversation')
#         if conversation:
#             lines.append("")
#             lines.append("💬 Nutritionist's Response:")
#             lines.append("─" * 60)
#             # Wrap conversation text at 60 characters
#             conv_words = conversation.split()
#             current_line = "  "
#             for word in conv_words:
#                 if len(current_line) + len(word) + 1 <= 62:
#                     current_line += word + " "
#                 else:
#                     lines.append(current_line.rstrip())
#                     current_line = "  " + word + " "
#             if current_line.strip():
#                 lines.append(current_line.rstrip())
#             lines.append("─" * 60)
#     else:
#         lines.append(f"LLM Analysis: ❌")
    
#     # Errors
#     if state.get("error"):
#         lines.append(f"❌ ERROR: {state['error']}")
#         lines.append(f"   Stage: {state.get('error_stage', 'unknown')}")
    
#     lines.append("="*60)
#     return "\n".join(lines)













# # """
# # LangGraph Workflows - FULLY FIXED Version
# # ==========================================
# # FIXES:
# # 1. ✅ All portions converted to grams (ml → g, oz → g, etc.)
# # 2. ✅ Total calories properly calculated from portions × database nutrition data
# # 3. ✅ Supports both image and text-only inputs
# # """
# # from typing import TypedDict, Dict, Any, Optional, List
# # from langgraph.graph import StateGraph, END
# # from PIL import Image
# # import io
# # import logging
# # import re

# # logger = logging.getLogger(__name__)


# # # ============================================================================
# # # STATE SCHEMA
# # # ============================================================================

# # class NutritionWorkflowState(TypedDict, total=False):
# #     """State for the nutrition analysis workflow."""
# #     # Input fields
# #     image: Optional[bytes]
# #     instruction: str
# #     system_prompt: Optional[str]
    
# #     # Vision analysis output
# #     raw_text: str
# #     parsed_json: Optional[Dict[str, Any]]
# #     portions: Optional[Dict[str, float]]  # ALL IN GRAMS
    
# #     # Database lookup output
# #     database_results: Optional[List[Dict[str, Any]]]
# #     milvus_results: Optional[List[Dict[str, Any]]]
    
# #     # LLM analysis output
# #     llm_analysis: Optional[Dict[str, Any]]
    
# #     # Final nutrition result
# #     nutrition_result: Optional[Dict[str, Any]]
    
# #     # Error handling
# #     error: Optional[str]
# #     error_stage: Optional[str]


# # # ============================================================================
# # # UNIT CONVERSION UTILITIES
# # # ============================================================================

# # def convert_to_grams(amount: float, unit: str) -> float:
# #     """
# #     Convert various units to grams.
    
# #     Args:
# #         amount: The numeric amount
# #         unit: The unit (g, ml, oz, lb, cup, tbsp, tsp, etc.)
    
# #     Returns:
# #         Amount in grams
# #     """
# #     unit = unit.lower().strip()
    
# #     # Already in grams
# #     if unit in ['g', 'gram', 'grams', 'gr']:
# #         return amount
    
# #     # Milliliters (assume density of water: 1ml = 1g)
# #     # For oils: 1ml ≈ 0.92g, but we'll use 1:1 for simplicity
# #     if unit in ['ml', 'milliliter', 'milliliters', 'mL']:
# #         return amount
    
# #     # Liters
# #     if unit in ['l', 'liter', 'liters', 'L']:
# #         return amount * 1000
    
# #     # Ounces (weight)
# #     if unit in ['oz', 'ounce', 'ounces']:
# #         return amount * 28.35
    
# #     # Pounds
# #     if unit in ['lb', 'lbs', 'pound', 'pounds']:
# #         return amount * 453.592
    
# #     # Cups (approximate, varies by ingredient)
# #     if unit in ['cup', 'cups', 'c']:
# #         return amount * 240  # Assume liquid cup
    
# #     # Tablespoons
# #     if unit in ['tbsp', 'tablespoon', 'tablespoons', 'T']:
# #         return amount * 15
    
# #     # Teaspoons
# #     if unit in ['tsp', 'teaspoon', 'teaspoons', 't']:
# #         return amount * 5
    
# #     # Kilograms
# #     if unit in ['kg', 'kilogram', 'kilograms']:
# #         return amount * 1000
    
# #     # If unknown unit, log warning and return original amount
# #     logger.warning(f"Unknown unit '{unit}', treating as grams")
# #     return amount


# # def parse_portion_string(portion_str: str) -> tuple[str, float]:
# #     """
# #     Parse a portion string like "chicken_breast:250g" or "olive_oil:15ml"
# #     and return (ingredient_name, grams).
    
# #     Args:
# #         portion_str: String in format "ingredient:amount_unit"
    
# #     Returns:
# #         Tuple of (ingredient_name, amount_in_grams)
# #     """
# #     try:
# #         if ':' not in portion_str:
# #             logger.warning(f"Invalid portion format (no colon): {portion_str}")
# #             return None, 0.0
        
# #         # Split into ingredient and amount
# #         ingredient, amount_str = portion_str.split(':', 1)
# #         ingredient = ingredient.strip()
# #         amount_str = amount_str.strip()
        
# #         # Extract number and unit using regex
# #         # Matches patterns like: "250g", "15ml", "2.5oz", "1.5 cup"
# #         match = re.match(r'(\d+(?:\.\d+)?)\s*([a-zA-Z]+)?', amount_str)
        
# #         if not match:
# #             logger.warning(f"Could not parse amount from: {amount_str}")
# #             return ingredient, 100.0  # Default 100g
        
# #         amount = float(match.group(1))
# #         unit = match.group(2) if match.group(2) else 'g'  # Default to grams if no unit
        
# #         # Convert to grams
# #         grams = convert_to_grams(amount, unit)
        
# #         logger.info(f"  Parsed: {ingredient} = {amount}{unit} → {grams}g")
# #         return ingredient, grams
        
# #     except Exception as e:
# #         logger.error(f"Error parsing portion '{portion_str}': {e}")
# #         return None, 0.0


# # # ============================================================================
# # # NODE 1: Vision Analysis (supports both image and text-only)
# # # ============================================================================

# # def vision_analysis_node(state: NutritionWorkflowState) -> Dict[str, Any]:
# #     """
# #     Analyze food using vision model (if image provided) or process text-only query.
# #     ALL PORTIONS RETURNED IN GRAMS.
# #     """
# #     import torch
# #     import json
    
# #     try:
# #         logger.info("🔍 Starting vision analysis...")
        
# #         # Get input data
# #         image_bytes = state.get("image")
# #         instruction = state.get("instruction", "Analyze the ingredients in this food.")
# #         system_prompt = state.get("system_prompt")
        
# #         # Force JSON output if not explicitly requesting it
# #         if "json" not in instruction.lower() and "JSON" not in instruction:
# #             if image_bytes:
# #                 json_instruction = f"""{instruction}

# # CRITICAL: Respond with ONLY a valid JSON object. No other text, explanations, or markdown. Start directly with {{ and end with }}.

# # Use this EXACT format (ALL AMOUNTS MUST INCLUDE UNITS):
# # {{
# #   "camera_or_phone_prob": <float 0.00-1.00>,
# #   "food_prob": <float 0.00-1.00>,
# #   "dish_name": "<lowercase camel case string>",
# #   "food_type": "<one of: home_cooked | restaurant_food | packaged | dessert | beverage | baked_goods>",
# #   "ingredients": ["<lowercase_snake_case strings>"],
# #   "portion_size": ["<item>:<amount><unit>"],
# #   "cooking_method": "<one of: frying | baking | grilling | boiling | steaming | roasting | raw | sauteing | stew>"
# # }}

# # IMPORTANT: In portion_size, ALWAYS include the unit (g, ml, oz, cup, tbsp, etc.)

# # Example:
# # {{
# #   "camera_or_phone_prob": 0.95,
# #   "food_prob": 0.98,
# #   "dish_name": "grilledChickenKebab",
# #   "food_type": "restaurant_food",
# #   "ingredients": ["chicken_breast", "olive_oil", "lemon", "onion"],
# #   "portion_size": ["chicken_breast:250g", "olive_oil:15ml", "lemon:20g", "onion:50g"],
# #   "cooking_method": "grilling"
# # }}"""
# #                 instruction = json_instruction
# #                 logger.info("✅ Added JSON formatting requirement to instruction")
        
# #         # ===================================================================
# #         # CASE 1: No image - text-only query
# #         # ===================================================================
# #         if not image_bytes:
# #             logger.info("📝 No image provided - text-only mode")
# #             logger.info(f"Processing instruction: {instruction}")
            
# #             from app.services.model_adapter import get_model_and_processor
# #             result = get_model_and_processor(has_image=False)
            
# #             if isinstance(result, tuple):
# #                 model, processor = result
# #                 logger.info("✅ Unpacked model and processor from tuple")
# #             elif isinstance(result, dict):
# #                 model = result.get('model')
# #                 processor = result.get('processor')
# #                 logger.info("✅ Extracted model and processor from dict")
# #             else:
# #                 raise ValueError(f"Unexpected return type from get_model_and_processor: {type(result)}")
            
# #             logger.info(f"✅ Model loaded for text-only: {type(model).__name__}")
            
# #             text_prompt_content = f"""Analyze this food description and extract ingredients with their estimated portions.

# # Food description: {instruction}

# # CRITICAL: You MUST respond with ONLY a valid JSON object. Do NOT include any other text, explanations, or markdown.

# # Use this EXACT format (INCLUDE UNITS):
# # {{
# #   "camera_or_phone_prob": <float>,
# #   "food_prob": <float>,
# #   "dish_name": "<lowercase camel case>",
# #   "food_type": "<type>",
# #   "ingredients": ["<ingredients>"],
# #   "portion_size": ["<item>:<amount><UNIT>"],
# #   "cooking_method": "<method>"
# # }}

# # IMPORTANT: Always include units in portion_size (g, ml, oz, cup, tbsp, tsp, etc.)"""
            
# #             messages = [
# #                 {
# #                     "role": "system",
# #                     "content": system_prompt or "You are a nutrition expert. Extract ingredients and portions from food descriptions."
# #                 },
# #                 {
# #                     "role": "user",
# #                     "content": text_prompt_content
# #                 }
# #             ]
            
# #             logger.info("🤖 Applying chat template for text-only analysis...")
# #             text_prompt = processor.apply_chat_template(
# #                 messages,
# #                 tokenize=False,
# #                 add_generation_prompt=True
# #             )
            
# #             # Get device
# #             if hasattr(model, 'device'):
# #                 device = model.device
# #             elif hasattr(model, 'model') and hasattr(model.model, 'device'):
# #                 device = model.model.device
# #             else:
# #                 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
# #             inputs = processor(
# #                 text=[text_prompt],
# #                 return_tensors="pt",
# #                 padding=True
# #             ).to(device)
            
# #             logger.info("🚀 Generating text-only analysis...")
# #             with torch.no_grad():
# #                 output_ids = model.generate(
# #                     **inputs,
# #                     max_new_tokens=512,
# #                     do_sample=True,
# #                     temperature=0.5,
# #                     top_p=0.9
# #                 )
            
# #             tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
# #             generated_text = tokenizer.batch_decode(
# #                 output_ids,
# #                 skip_special_tokens=True,
# #                 clean_up_tokenization_spaces=True
# #             )[0]
            
# #             response_text = generated_text.split("assistant\n")[-1].strip()
            
# #             del inputs, output_ids
# #             if device.type == "cuda":
# #                 torch.cuda.empty_cache()
        
# #         # ===================================================================
# #         # CASE 2: Image provided - vision analysis
# #         # ===================================================================
# #         else:
# #             logger.info("📦 Converting bytes to PIL Image...")
# #             image = Image.open(io.BytesIO(image_bytes))
# #             if image.mode != "RGB":
# #                 image = image.convert("RGB")
# #             logger.info(f"✅ Image converted: {image.size}, mode: {image.mode}")
            
# #             from app.services.model_adapter import get_model_and_processor
# #             result = get_model_and_processor(has_image=True)
            
# #             if isinstance(result, tuple):
# #                 model, processor = result
# #                 logger.info("✅ Unpacked model and processor from tuple")
# #             elif isinstance(result, dict):
# #                 model = result.get('model')
# #                 processor = result.get('processor')
# #                 logger.info("✅ Extracted model and processor from dict")
# #             else:
# #                 raise ValueError(f"Unexpected return type: {type(result)}")
            
# #             logger.info(f"✅ Model loaded: {type(model).__name__}")
            
# #             conversation = []
# #             if system_prompt:
# #                 conversation.append({"role": "system", "content": system_prompt})
            
# #             conversation.append({
# #                 "role": "user",
# #                 "content": [
# #                     {"type": "image", "image": image},
# #                     {"type": "text", "text": instruction}
# #                 ]
# #             })
            
# #             logger.info("🤖 Applying chat template...")
# #             text_prompt = processor.apply_chat_template(
# #                 conversation,
# #                 tokenize=False,
# #                 add_generation_prompt=True
# #             )
            
# #             if hasattr(model, 'device'):
# #                 device = model.device
# #             elif hasattr(model, 'model') and hasattr(model.model, 'device'):
# #                 device = model.model.device
# #             else:
# #                 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
# #             logger.info("📝 Processing inputs...")
# #             inputs = processor(
# #                 text=[text_prompt],
# #                 images=[image],
# #                 return_tensors="pt",
# #                 padding=True
# #             ).to(device)
            
# #             logger.info("🚀 Generating vision analysis...")
# #             with torch.no_grad():
# #                 output_ids = model.generate(
# #                     **inputs,
# #                     max_new_tokens=1024,
# #                     do_sample=True,
# #                     temperature=0.5,
# #                     top_p=0.9
# #                 )
            
# #             logger.info("📤 Decoding output...")
# #             generated_text = processor.batch_decode(
# #                 output_ids,
# #                 skip_special_tokens=True,
# #                 clean_up_tokenization_spaces=True
# #             )[0]
            
# #             response_text = generated_text.split("assistant\n")[-1].strip()
            
# #             del inputs, output_ids
# #             if torch.cuda.is_available():
# #                 torch.cuda.empty_cache()
        
# #         # ===================================================================
# #         # Common processing - Parse JSON and extract portions IN GRAMS
# #         # ===================================================================
        
# #         logger.info(f"✅ Generated {len(response_text)} characters")
# #         logger.info(f"Raw response preview: {response_text[:300]}...")
        
# #         # Parse JSON from response
# #         json_str = None
# #         parsed_json = None
        
# #         # Try multiple extraction methods
# #         json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
# #         if json_match:
# #             json_str = json_match.group(1)
# #             logger.info("✅ Found JSON in markdown code block")
        
# #         if not json_str:
# #             json_match = re.search(r'```\s*(\{.*?\})\s*```', response_text, re.DOTALL)
# #             if json_match:
# #                 json_str = json_match.group(1)
# #                 logger.info("✅ Found JSON in plain code block")
        
# #         if not json_str:
# #             start = response_text.find('{')
# #             end = response_text.rfind('}')
# #             if start != -1 and end != -1 and start < end:
# #                 json_str = response_text[start:end+1]
# #                 logger.info("✅ Extracted JSON from { to }")
        
# #         if not json_str:
# #             logger.warning("⚠️ No JSON delimiters found, trying entire response")
# #             json_str = response_text
        
# #         try:
# #             parsed_json = json.loads(json_str)
# #             logger.info("✅ JSON parsed successfully")
# #         except json.JSONDecodeError as e:
# #             logger.error(f"❌ JSON parsing failed: {e}")
# #             logger.info("🔧 Attempting JSON cleanup...")
# #             try:
# #                 cleaned = json_str[json_str.find('{'):]
# #                 cleaned = cleaned[:cleaned.rfind('}')+1]
# #                 cleaned = cleaned.replace('**', '')
# #                 parsed_json = json.loads(cleaned)
# #                 logger.info("✅ JSON parsed after cleanup")
# #             except:
# #                 logger.error("❌ JSON cleanup failed")
# #                 return {
# #                     "error": f"Failed to parse JSON: {e}",
# #                     "error_stage": "vision_analysis",
# #                     "raw_text": response_text
# #                 }
        
# #         # =================================================================
# #         # Extract portions and CONVERT ALL TO GRAMS
# #         # =================================================================
# #         logger.info("📊 Parsing portions and converting to grams...")
# #         portions = {}
        
# #         portion_size_list = parsed_json.get("portion_size", [])
# #         if portion_size_list:
# #             for portion_str in portion_size_list:
# #                 ingredient, grams = parse_portion_string(portion_str)
# #                 if ingredient and grams > 0:
# #                     portions[ingredient] = grams
        
# #         # Fallback: use ingredients with default 100g
# #         if not portions:
# #             logger.warning("⚠️ No portion_size field, using default 100g per ingredient")
# #             ingredients_list = parsed_json.get("ingredients", [])
# #             for ing in ingredients_list:
# #                 portions[ing] = 100.0
# #                 logger.info(f"  - {ing}: 100g (default)")
        
# #         logger.info(f"✅ Extracted {len(portions)} ingredients (ALL IN GRAMS):")
# #         for name, grams in portions.items():
# #             logger.info(f"  - {name}: {grams}g")
        
# #         # Log additional fields
# #         camera_prob = parsed_json.get("camera_or_phone_prob", 0.0)
# #         food_prob = parsed_json.get("food_prob", 0.0)
# #         food_type = parsed_json.get("food_type", "unknown")
# #         cooking_method = parsed_json.get("cooking_method", "unknown")
        
# #         logger.info(f"📸 Camera probability: {camera_prob:.2f}")
# #         logger.info(f"🍽️  Food probability: {food_prob:.2f}")
# #         logger.info(f"📋 Food type: {food_type}")
# #         logger.info(f"🔥 Cooking method: {cooking_method}")
        
# #         # Add portions to parsed_json
# #         parsed_json["portions"] = portions
        
# #         return {
# #             "raw_text": response_text,
# #             "parsed_json": parsed_json,
# #             "portions": portions  # ALL IN GRAMS
# #         }
        
# #     except Exception as e:
# #         logger.error(f"❌ Vision analysis failed: {e}", exc_info=True)
# #         return {
# #             "error": str(e),
# #             "error_stage": "vision_analysis"
# #         }


# # # ============================================================================
# # # NODE 2: Database Lookup
# # # ============================================================================

# # def database_lookup_node(state: NutritionWorkflowState) -> Dict[str, Any]:
# #     """Search database for top-3 matches for each ingredient."""
# #     from app.services.milvus_service import MilvusManager
    
# #     try:
# #         logger.info("🔍 Starting database lookup...")
        
# #         portions = state.get("portions", {})
        
# #         if not portions:
# #             logger.warning("No portions found, skipping database lookup")
# #             return {
# #                 "database_results": [],
# #                 "milvus_results": []
# #             }
        
# #         ingredient_names = list(portions.keys())
# #         logger.info(f"Looking up {len(ingredient_names)} ingredients: {ingredient_names}")
        
# #         # Clean ingredient names
# #         cleaned_ingredient_names = [
# #             name.replace('_', ' ').replace('-', ' ') 
# #             for name in ingredient_names
# #         ]
# #         logger.info(f"Cleaned names for search: {cleaned_ingredient_names}")
        
# #         milvus = MilvusManager.get_instance()
        
# #         try:
# #             results = milvus.search_ingredients(cleaned_ingredient_names, top_k=3)
# #             logger.info(f"✅ Found {len(results)} ingredient matches")
            
# #             for result in results:
# #                 query = result.get("query", "Unknown")
# #                 matches = result.get("results", [])
# #                 logger.info(f"  {query}: {len(matches)} matches")
# #                 if matches:
# #                     best = matches[0]  # The match IS the entity
# #                     logger.info(f"    Best: {best.get('item_name', 'N/A')} (score: {best.get('score', 0):.3f})")
            
# #             return {
# #                 "database_results": results,
# #                 "milvus_results": results
# #             }
            
# #         except Exception as e:
# #             logger.error(f"Milvus search failed: {e}", exc_info=True)
# #             return {
# #                 "database_results": [],
# #                 "milvus_results": [],
# #                 "error": f"Database search failed: {e}",
# #                 "error_stage": "database_lookup"
# #             }
        
# #     except Exception as e:
# #         logger.error(f"❌ Database lookup failed: {e}", exc_info=True)
# #         return {
# #             "error": str(e),
# #             "error_stage": "database_lookup"
# #         }


# # # ============================================================================
# # # NODE 3: LLM Analysis with PROPER CALORIE CALCULATION
# # # ============================================================================

# # def llm_analysis_node(state: NutritionWorkflowState) -> Dict[str, Any]:
# #     """
# #     Analyze nutrition using LLM with PROPER CALORIE CALCULATION.
    
# #     Key fix: Multiply portion size (grams) by nutritional values per 100g,
# #     then sum all ingredients to get total calories.
# #     """
# #     try:
# #         logger.info("🤖 Starting LLM analysis with calorie calculation...")
        
# #         portions = state.get("portions", {})
# #         database_results = state.get("database_results", [])
# #         parsed_json = state.get("parsed_json", {})
        
# #         if not portions:
# #             logger.warning("No portions to analyze")
# #             return {"llm_analysis": None}
        
# #         if not database_results:
# #             logger.warning("No database results to calculate nutrition")
# #             return {"llm_analysis": None}
        
# #         # =================================================================
# #         # CALCULATE NUTRITION FOR EACH INGREDIENT
# #         # =================================================================
# #         logger.info("📊 Calculating nutrition for each ingredient...")
        
# #         ingredient_nutrition = []
# #         total_calories = 0.0
# #         total_protein = 0.0
# #         total_carbs = 0.0
# #         total_fat = 0.0
# #         total_fiber = 0.0
        
# #         for db_result in database_results:
# #             query = db_result.get("query", "")
# #             matches = db_result.get("results", [])
            
# #             if not matches:
# #                 logger.warning(f"No matches for: {query}")
# #                 continue
            
# #             # Get best match (highest score)
# #             best_match = matches[0]
# #             entity = best_match  # The match object IS the entity
# #             score = entity.get("score", 0.0)
            
# #             # Get nutrition data (per 100g from database)
# #             # NOTE: Database uses specific field names
# #             food_name = entity.get("item_name", entity.get("name", query))
# #             calories_per_100g = entity.get("calories", 0.0)
# #             protein_per_100g = entity.get("protein_g", 0.0)
# #             carbs_per_100g = entity.get("carb_g", 0.0)
# #             fat_per_100g = entity.get("fat_g", 0.0)
# #             fiber_per_100g = entity.get("fiber_g", 0.0)
            
# #             # Find corresponding portion (in grams)
# #             # Query might be cleaned, so try both versions
# #             portion_grams = 0.0
            
# #             # Try exact match first
# #             if query in portions:
# #                 portion_grams = portions[query]
# #             else:
# #                 # Try to find by matching cleaned names
# #                 query_cleaned = query.replace(' ', '_').replace('-', '_')
# #                 for ing_name, ing_grams in portions.items():
# #                     ing_cleaned = ing_name.replace(' ', '_').replace('-', '_')
# #                     if query_cleaned == ing_cleaned or query in ing_name or ing_name in query:
# #                         portion_grams = ing_grams
# #                         break
            
# #             if portion_grams == 0:
# #                 logger.warning(f"Could not find portion for '{query}' in portions: {list(portions.keys())}")
# #                 continue
            
# #             # CALCULATE actual nutrition based on portion size
# #             # Formula: (nutrition_per_100g * portion_grams) / 100
# #             multiplier = portion_grams / 100.0
            
# #             actual_calories = calories_per_100g * multiplier
# #             actual_protein = protein_per_100g * multiplier
# #             actual_carbs = carbs_per_100g * multiplier
# #             actual_fat = fat_per_100g * multiplier
# #             actual_fiber = fiber_per_100g * multiplier
            
# #             logger.info(f"  {query} ({portion_grams}g):")
# #             logger.info(f"    DB: {calories_per_100g} cal/100g → Actual: {actual_calories:.1f} cal")
# #             logger.info(f"    Protein: {actual_protein:.1f}g | Carbs: {actual_carbs:.1f}g | Fat: {actual_fat:.1f}g")
            
# #             # Add to ingredient list
# #             ingredient_nutrition.append({
# #                 "name": food_name,
# #                 "portion_grams": portion_grams,
# #                 "match_score": score,
# #                 "calories": actual_calories,
# #                 "protein": actual_protein,
# #                 "carbohydrates": actual_carbs,
# #                 "fat": actual_fat,
# #                 "fiber": actual_fiber,
# #                 # Include per-100g values for reference
# #                 "per_100g": {
# #                     "calories": calories_per_100g,
# #                     "protein": protein_per_100g,
# #                     "carbohydrates": carbs_per_100g,
# #                     "fat": fat_per_100g,
# #                     "fiber": fiber_per_100g
# #                 }
# #             })
            
# #             # Sum totals
# #             total_calories += actual_calories
# #             total_protein += actual_protein
# #             total_carbs += actual_carbs
# #             total_fat += actual_fat
# #             total_fiber += actual_fiber
        
# #         # =================================================================
# #         # BUILD FINAL ANALYSIS
# #         # =================================================================
# #         logger.info(f"✅ Calculated totals:")
# #         logger.info(f"  Total calories: {total_calories:.1f}")
# #         logger.info(f"  Total protein: {total_protein:.1f}g")
# #         logger.info(f"  Total carbs: {total_carbs:.1f}g")
# #         logger.info(f"  Total fat: {total_fat:.1f}g")
# #         logger.info(f"  Total fiber: {total_fiber:.1f}g")
        
# #         llm_analysis = {
# #             "dish_name": parsed_json.get("dish_name", "Unknown Dish"),
# #             "food_type": parsed_json.get("food_type", "unknown"),
# #             "cooking_method": parsed_json.get("cooking_method", "unknown"),
# #             "ingredients": ingredient_nutrition,
# #             "total_nutrition": {
# #                 "calories": round(total_calories, 1),
# #                 "protein": round(total_protein, 1),
# #                 "carbohydrates": round(total_carbs, 1),
# #                 "fat": round(total_fat, 1),
# #                 "fiber": round(total_fiber, 1)
# #             },
# #             "metadata": {
# #                 "total_ingredients": len(ingredient_nutrition),
# #                 "camera_prob": parsed_json.get("camera_or_phone_prob", 0.0),
# #                 "food_prob": parsed_json.get("food_prob", 0.0)
# #             }
# #         }
        
# #         logger.info("✅ LLM analysis completed with proper calorie calculation")
        
# #         return {
# #             "llm_analysis": llm_analysis,
# #             "nutrition_result": llm_analysis
# #         }
        
# #     except Exception as e:
# #         logger.error(f"❌ LLM analysis failed: {e}", exc_info=True)
# #         return {
# #             "error": str(e),
# #             "error_stage": "llm_analysis"
# #         }


# # # ============================================================================
# # # WORKFLOW BUILDER
# # # ============================================================================

# # def build_nutrition_workflow():
# #     """
# #     Build the complete nutrition analysis workflow.
# #     Supports both image and text-only inputs.
# #     """
# #     logger.info("Building nutrition workflow with fixed calorie calculation...")
    
# #     workflow = StateGraph(NutritionWorkflowState)
    
# #     # Add nodes
# #     workflow.add_node("vision_analysis", vision_analysis_node)
# #     workflow.add_node("database_lookup", database_lookup_node)
# #     workflow.add_node("llm_analysis", llm_analysis_node)
    
# #     # Define edges
# #     workflow.add_edge("vision_analysis", "database_lookup")
# #     workflow.add_edge("database_lookup", "llm_analysis")
# #     workflow.add_edge("llm_analysis", END)
    
# #     # Set entry point
# #     workflow.set_entry_point("vision_analysis")
    
# #     logger.info("✅ Workflow structure: vision_analysis → database_lookup → llm_analysis → END")
    
# #     return workflow.compile()


# # # ============================================================================
# # # UTILITY FUNCTIONS
# # # ============================================================================

# # def validate_workflow_state(state: NutritionWorkflowState) -> bool:
# #     """Check if workflow state is valid."""
# #     if state.get("error"):
# #         logger.error(f"Workflow error at {state.get('error_stage', 'unknown')}: {state['error']}")
# #         return False
# #     return True


# # def get_workflow_summary(state: NutritionWorkflowState) -> str:
# #     """Get a detailed summary of the workflow state."""
# #     lines = []
# #     lines.append("="*60)
# #     lines.append("WORKFLOW STATE SUMMARY")
# #     lines.append("="*60)
    
# #     # Input
# #     has_image = "✅" if state.get("image") else "❌"
# #     lines.append(f"Image: {has_image}")
# #     lines.append(f"Instruction: {state.get('instruction', 'None')[:50]}...")
    
# #     # Vision analysis
# #     has_raw = "✅" if state.get("raw_text") else "❌"
# #     has_parsed = "✅" if state.get("parsed_json") else "❌"
# #     portions = state.get("portions", {})
# #     lines.append(f"Raw text: {has_raw} | Parsed JSON: {has_parsed}")
# #     lines.append(f"Portions: {len(portions)} ingredients (all in grams)")
    
# #     if portions:
# #         lines.append("  Ingredient portions:")
# #         for name, grams in portions.items():
# #             lines.append(f"    - {name}: {grams}g")
    
# #     # Database lookup
# #     db_results = state.get("milvus_results", [])
# #     lines.append(f"Database results: {len(db_results)} matches")
    
# #     # LLM analysis
# #     llm_analysis = state.get("llm_analysis")
# #     if llm_analysis:
# #         lines.append(f"LLM Analysis: ✅")
# #         lines.append(f"  Dish: {llm_analysis.get('dish_name', 'Unknown')}")
        
# #         total_nutrition = llm_analysis.get('total_nutrition', {})
# #         calories = total_nutrition.get('calories', 0)
# #         protein = total_nutrition.get('protein', 0)
# #         carbs = total_nutrition.get('carbohydrates', 0)
# #         fat = total_nutrition.get('fat', 0)
        
# #         lines.append(f"  Total Calories: {calories:.1f} kcal")
# #         lines.append(f"  Protein: {protein:.1f}g | Carbs: {carbs:.1f}g | Fat: {fat:.1f}g")
        
# #         # Show ingredient breakdown
# #         ingredients = llm_analysis.get('ingredients', [])
# #         if ingredients:
# #             lines.append("  Ingredient breakdown:")
# #             for ing in ingredients:
# #                 name = ing.get('name', 'Unknown')
# #                 portion = ing.get('portion_grams', 0)
# #                 cals = ing.get('calories', 0)
# #                 lines.append(f"    - {name} ({portion}g): {cals:.1f} cal")
# #     else:
# #         lines.append(f"LLM Analysis: ❌")
    
# #     # Errors
# #     if state.get("error"):
# #         lines.append(f"❌ ERROR: {state['error']}")
# #         lines.append(f"   Stage: {state.get('error_stage', 'unknown')}")
    
# #     lines.append("="*60)
# #     return "\n".join(lines)