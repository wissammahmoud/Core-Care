"""
LLM Analysis Node - Fixed String Formatting
============================================
Fixes KeyError by properly escaping JSON examples in prompt templates.
"""
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


# Prompt template with ESCAPED curly braces for JSON examples
# Use {{ and }} to represent literal { and } in the output
LLM_ANALYSIS_PROMPT = """You are a nutrition analysis assistant. Analyze the following food data and provide detailed nutritional information.

Food Information:
- Dish Name: {dish_name}
- Ingredients: {ingredients}

Database Results:
{database_info}

Milvus Search Results (similar foods):
{milvus_info}

Instructions:
1. Analyze the nutritional content based on the ingredients and portions
2. Use the database results to get accurate nutritional values
3. Consider the Milvus results for similar foods to enhance your analysis
4. Provide a comprehensive nutritional breakdown
5. Include recommendations if applicable

Respond with ONLY a valid JSON object in this format (no other text):
{{{{
  "total_calories": number,
  "macros": {{{{
    "protein_g": number,
    "carbs_g": number,
    "fat_g": number,
    "fiber_g": number
  }}}},
  "micronutrients": {{{{
    "vitamin_a_mcg": number,
    "vitamin_c_mg": number,
    "calcium_mg": number,
    "iron_mg": number
  }}}},
  "health_notes": "brief health assessment",
  "recommendations": "dietary recommendations if applicable"
}}}}

Remember: ONLY JSON, no other text!"""


def llm_analysis_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform LLM analysis on the nutrition data.
    
    Args:
        state: Workflow state containing parsed_json, database_results, milvus_results
    
    Returns:
        Dict with llm_analysis and nutrition_result
    """
    import json
    import re
    
    try:
        logger.info("🤖 Starting LLM analysis...")
        
        # Get data from state
        parsed_json = state.get("parsed_json")
        database_results = state.get("database_results", [])
        milvus_results = state.get("milvus_results", [])
        
        if not parsed_json:
            logger.error("❌ No parsed_json in state")
            return {
                "error": "No parsed JSON data available for LLM analysis",
                "error_stage": "llm_analysis"
            }
        
        # Extract dish name and ingredients
        dish_name = parsed_json.get("dish_name", "Unknown dish")
        ingredients_list = parsed_json.get("ingredients", [])
        
        # Format ingredients string - handle both formats
        if ingredients_list:
            formatted_ingredients = []
            for ing in ingredients_list:
                if isinstance(ing, dict):
                    # Old format: {"name": "...", "grams": 100}
                    name = ing.get('name', 'unknown')
                    grams = ing.get('grams', 0)
                    formatted_ingredients.append(f"{name} ({grams}g)")
                elif isinstance(ing, str):
                    # New format: just string names
                    # Try to get portion from portions dict if available
                    portions = parsed_json.get("portions", {})
                    if ing in portions:
                        grams = portions[ing]
                        formatted_ingredients.append(f"{ing} ({grams}g)")
                    else:
                        # No portion info, just use name
                        formatted_ingredients.append(ing)
            
            ingredients_str = ", ".join(formatted_ingredients)
        else:
            ingredients_str = "No ingredients specified"
        
        logger.info(f"Analyzing: {dish_name}")
        logger.info(f"Ingredients: {ingredients_str}")
        
        # Format database results
        if database_results:
            db_info_lines = []
            for result in database_results[:5]:  # Top 5
                name = result.get("name", "Unknown")
                calories = result.get("calories", 0)
                protein = result.get("protein_g", 0)
                db_info_lines.append(f"  - {name}: {calories} cal, {protein}g protein")
            database_info = "Database matches:\n" + "\n".join(db_info_lines)
        else:
            database_info = "No database matches found"
        
        # Format Milvus results
        if milvus_results:
            milvus_info_lines = []
            for result in milvus_results[:3]:  # Top 3
                name = result.get("name", "Unknown")
                similarity = result.get("similarity", 0)
                milvus_info_lines.append(f"  - {name} (similarity: {similarity:.2f})")
            milvus_info = "Similar foods:\n" + "\n".join(milvus_info_lines)
        else:
            milvus_info = "No similar foods found"
        
        # Build the full prompt
        logger.info("📝 Building LLM prompt...")
        full_prompt = LLM_ANALYSIS_PROMPT.format(
            dish_name=dish_name,
            ingredients=ingredients_str,
            database_info=database_info,
            milvus_info=milvus_info
        )
        
        logger.info(f"Prompt length: {len(full_prompt)} characters")
        
        # Get model and processor
        from app.services.model_adapter import get_model_and_processor
        
        logger.info("Loading model for LLM analysis...")
        result = get_model_and_processor(has_image=False)
        
        if isinstance(result, tuple):
            model, processor = result
        elif isinstance(result, dict):
            model = result.get('model')
            processor = result.get('processor')
        else:
            raise ValueError(f"Unexpected return type: {type(result)}")
        
        logger.info(f"✅ Model loaded: {type(model).__name__}")
        
        # Prepare conversation
        conversation = [
            {
                "role": "user",
                "content": full_prompt
            }
        ]
        
        # Apply chat template
        logger.info("🤖 Applying chat template...")
        text_prompt = processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Get device
        import torch
        if hasattr(model, 'device'):
            device = model.device
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Process inputs
        logger.info("📝 Processing inputs...")
        inputs = processor(
            text=[text_prompt],
            return_tensors="pt",
            padding=True
        ).to(device)
        
        # Generate
        logger.info("🚀 Generating LLM analysis...")
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
        
        # Decode
        logger.info("📤 Decoding output...")
        tokenizer = processor.tokenizer if hasattr(processor, 'tokenizer') else processor
        generated_text = tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0]
        
        # Extract response
        response_text = generated_text.split("assistant\n")[-1].strip()
        
        # Cleanup
        del inputs, output_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info(f"✅ Generated {len(response_text)} characters")
        logger.info(f"Response preview: {response_text[:200]}...")
        
        # Parse JSON from response
        json_str = None
        
        # Try different methods
        json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            logger.info("✅ Found JSON in markdown block")
        
        if not json_str:
            start = response_text.find('{')
            end = response_text.rfind('}')
            if start != -1 and end != -1:
                json_str = response_text[start:end+1]
                logger.info("✅ Extracted JSON from { to }")
        
        if not json_str:
            json_str = response_text
            logger.warning("⚠️ Using entire response as JSON")
        
        # Parse JSON
        try:
            llm_analysis = json.loads(json_str)
            logger.info("✅ LLM analysis JSON parsed successfully")
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON parsing failed: {e}")
            logger.error(f"Attempted to parse: {json_str[:500]}")
            
            # Try cleanup
            try:
                cleaned = json_str[json_str.find('{'):json_str.rfind('}')+1]
                cleaned = cleaned.replace('**', '')
                llm_analysis = json.loads(cleaned)
                logger.info("✅ JSON parsed after cleanup")
            except:
                logger.error("❌ Cleanup also failed")
                return {
                    "error": f"Failed to parse LLM response JSON: {e}",
                    "error_stage": "llm_analysis",
                    "raw_llm_response": response_text
                }
        
        # Build final nutrition result
        nutrition_result = {
            "dish_name": dish_name,
            "ingredients": ingredients_list,
            "analysis": llm_analysis,
            "database_matches": len(database_results),
            "similar_foods": len(milvus_results)
        }
        
        logger.info("✅ LLM analysis completed successfully!")
        
        return {
            "llm_analysis": llm_analysis,
            "nutrition_result": nutrition_result
        }
        
    except Exception as e:
        logger.error(f"❌ LLM analysis node failed: {e}", exc_info=True)
        return {
            "error": str(e),
            "error_stage": "llm_analysis"   
        }