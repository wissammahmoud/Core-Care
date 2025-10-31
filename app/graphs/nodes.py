"""
LangGraph node implementations with LoRA support and enhanced debugging.
"""
from app.graphs.states import NutritionState
from app.services.model_service import ModelManager
from app.services.milvus_service import MilvusManager
from app.utils.image_utils import load_and_resize_image
from app.utils.parsers import extract_portions_from_json, extract_json_from_text
from app.models.prompt import SYSTEM_PROMPT
import torch
import logging
import os

logger = logging.getLogger(__name__)


def load_image_node(state: NutritionState) -> NutritionState:
    """
    Load and preprocess image (if provided).
    """
    logger.info("Node: load_image")
    
    try:
        # Check if image bytes provided
        if not state.get("image_bytes"):
            logger.info("No image provided - text-only mode")
            return {"has_image": False, "error": None}
        
        # If image already loaded, skip
        if state.get("image"):
            logger.info("Image already loaded, returning has_image=True")
            return {"has_image": True, "error": None}
        
        # Load and resize image
        img = load_and_resize_image(
            state["image_bytes"],
            max_size=int(os.getenv('MAX_IMAGE_SIZE', '448'))
        )
        
        logger.info(f"✅ Image loaded successfully: {img.size}")
        logger.info(f"✅ Setting has_image=True in state")
        return {"image": img, "has_image": True, "error": None}
    
    except Exception as e:
        logger.error(f"❌ Image loading failed: {e}")
        return {"error": str(e), "error_stage": "load_image"}


def inference_node(state: NutritionState) -> NutritionState:
    """
    Run VLM inference with enhanced debugging.
    """
    logger.info("Node: inference")
    
    # DEBUG: Show state at start of inference
    logger.info("="*60)
    logger.info("DEBUG: State at inference start:")
    logger.info(f"  has_image in state: {state.get('has_image')}")
    logger.info(f"  image in state: {state.get('image') is not None}")
    logger.info(f"  instruction: {state.get('instruction', '')[:50]}...")
    logger.info("="*60)
    
    if state.get("error"):
        logger.warning("Skipping inference due to previous error")
        return {}
    
    try:
        # Get model manager
        model_mgr = ModelManager.get_instance()
        
        # Determine if this is an image or text-only query
        has_image = state.get("has_image", False) and state.get("image") is not None
        
        logger.info(f"🔍 has_image determined as: {has_image}")
        logger.info(f"   - state.get('has_image'): {state.get('has_image')}")
        logger.info(f"   - state.get('image') is not None: {state.get('image') is not None}")
        
        # Initialize models if not already loaded
        base_model_id = os.getenv('MODEL_ID', 'unsloth/Qwen3-VL-4B-Instruct-bnb-4bit')
        lora_adapter_id = os.getenv('LORA_ADAPTER_ID', 'WissMah/Qwen2.5VL-FT-Lora_mix_aug1')
        
        if model_mgr.base_model is None:
            logger.info("Initializing models...")
            model_mgr.initialize_models(base_model_id, lora_adapter_id)
        
        # Get appropriate model based on input type
        model, processor, tokenizer = model_mgr.get_model_for_input(has_image)
        
        # Get system prompt
        system_prompt = state.get("system_prompt", SYSTEM_PROMPT)
        
        # Prepare messages
        messages = []
        
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        if has_image:
            # Image input
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image", "image": state["image"]},
                    {"type": "text", "text": state["instruction"]},
                ],
            })
            logger.info("✅ Processing IMAGE + TEXT input")
        else:
            # Text-only input
            messages.append({
                "role": "user",
                "content": state["instruction"]
            })
            logger.info("⚠️ Processing TEXT-ONLY input (NO IMAGE)")
        
        logger.info("Applying chat template...")
        text_prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Show prompt preview
        logger.info(f"📝 Prompt preview (first 200 chars): {text_prompt[:200]}...")
        
        # Process inputs
        device = next(model.parameters()).device
        
        if has_image:
            logger.info("🖼️ Tokenizing with image...")
            inputs = processor(
                text=[text_prompt],
                images=[state["image"]],
                return_tensors="pt",
                padding=True,
            ).to(device)
        else:
            logger.info("📄 Tokenizing text only...")
            inputs = processor(
                text=[text_prompt],
                return_tensors="pt",
                padding=True,
            ).to(device)
        
        max_tokens = int(os.getenv('MAX_NEW_TOKENS', '256'))
        logger.info(f"🚀 Running inference (max {max_tokens} tokens)...")
        
        # Generate
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=0.0,
                use_cache=True,
            )
        
        # Decode
        prompt_len = inputs["input_ids"].shape[-1]
        generated_ids = output_ids[0, prompt_len:]
        result = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        # Cleanup
        model_mgr.cleanup()
        
        # DEBUG: Show raw output
        logger.info("="*60)
        logger.info("🎯 RAW MODEL OUTPUT:")
        logger.info(f"Length: {len(result)} characters")
        logger.info("-"*60)
        logger.info(result)
        logger.info("="*60)
        
        return {"raw_text": result, "error": None}
    
    except Exception as e:
        logger.error(f"❌ Inference failed: {e}", exc_info=True)
        return {"error": str(e), "error_stage": "inference"}


def parse_node(state: NutritionState) -> NutritionState:
    """
    Parse JSON with enhanced debugging.
    """
    logger.info("Node: parse")
    
    if state.get("error"):
        return {}
    
    try:
        raw_text = state.get("raw_text", "")
        
        if not raw_text:
            logger.error("❌ Empty response from model!")
            return {"error": "Empty response from model", "error_stage": "parse"}
        
        logger.info(f"📊 Parsing {len(raw_text)} characters...")
        logger.info(f"Raw text preview: {raw_text[:100]}...")
        
        # Try to extract JSON from text
        parsed = extract_json_from_text(raw_text)
        
        logger.info(f"✅ JSON extracted: {parsed}")
        
        # Extract portions
        portions = extract_portions_from_json(parsed)
        
        logger.info(f"✅ Extracted {len(portions)} ingredients")
        if portions:
            logger.info(f"Portions: {portions}")
        
        return {
            "parsed_json": parsed,
            "portions": portions,
            "error": None
        }
    
    except Exception as e:
        logger.error(f"❌ JSON parsing failed: {e}", exc_info=True)
        logger.error(f"Raw text that failed: {raw_text}")
        return {"error": f"Failed to parse JSON: {str(e)}", "error_stage": "parse"}


def database_lookup_node(state: NutritionState) -> NutritionState:
    """
    Look up ingredients in Milvus.
    """
    logger.info("Node: database_lookup")
    
    if state.get("error"):
        return {}
    
    try:
        portions = state.get("portions", {})
        
        if not portions:
            logger.warning("⚠️ No portions found, skipping database lookup")
            return {"milvus_results": [], "error": None}
        
        ingredient_names = list(portions.keys())
        logger.info(f"🔍 Looking up {len(ingredient_names)} ingredients: {ingredient_names}")
        
        # Get Milvus manager
        milvus_mgr = MilvusManager.get_instance()
        
        # Search for ingredients
        results = milvus_mgr.search_ingredients(
            ingredient_names,
            top_k=3
        )
        
        logger.info(f"✅ Found {len(results)} ingredient matches")
        
        return {"milvus_results": results, "error": None}
    
    except Exception as e:
        logger.error(f"❌ Database lookup failed: {e}", exc_info=True)
        return {"error": str(e), "error_stage": "database_lookup"}


def calculate_nutrition_node(state: NutritionState) -> NutritionState:
    """
    Calculate total nutrition.
    """
    logger.info("Node: calculate_nutrition")
    
    if state.get("error"):
        return {}
    
    try:
        portions = state.get("portions", {})
        db_results = state.get("milvus_results", [])
        
        if not portions:
            logger.warning("⚠️ No portions to calculate")
            return {
                "nutrition_result": {
                    "per_ingredient": [],
                    "totals": {"grams": 0, "calories": 0, "protein": 0, "carb": 0, "fat": 0},
                    "missing_ingredients": []
                },
                "error": None
            }
        
        # Map ingredients to macros
        macros_map = {}
        for result in db_results:
            query = result["query"]
            best_match = result["results"][0] if result["results"] else None
            if best_match:
                macros_map[query] = best_match
        
        # Calculate per-ingredient and totals
        per_ingredient = []
        totals = {"grams": 0.0, "calories": 0.0, "protein": 0.0, "carb": 0.0, "fat": 0.0}
        missing = []
        
        for ingredient, grams in portions.items():
            grams = float(grams)
            totals["grams"] += grams
            
            macro = macros_map.get(ingredient)
            if not macro:
                logger.warning(f"⚠️ No macro data found for: {ingredient}")
                missing.append(ingredient)
                continue
            
            # Scale to grams (database values are per 100g)
            factor = grams / 100.0
            
            per_ingredient.append({
                "ingredient": ingredient,
                "grams": round(grams, 2),
                "calories": round(macro["calories"] * factor, 2),
                "protein": round(macro["protein_g"] * factor, 2),
                "carb": round(macro["carb_g"] * factor, 2),
                "fat": round(macro["fat_g"] * factor, 2),
            })
            
            totals["calories"] += macro["calories"] * factor
            totals["protein"] += macro["protein_g"] * factor
            totals["carb"] += macro["carb_g"] * factor
            totals["fat"] += macro["fat_g"] * factor
        
        # Round totals
        for key in totals:
            totals[key] = round(totals[key], 2)
        
        result = {
            "per_ingredient": per_ingredient,
            "totals": totals,
            "missing_ingredients": missing,
        }
        
        logger.info(f"✅ Calculated nutrition: {totals['calories']} kcal total")
        
        return {"nutrition_result": result, "error": None}
    
    except Exception as e:
        logger.error(f"❌ Calculation failed: {e}", exc_info=True)
        return {"error": str(e), "error_stage": "calculate"}