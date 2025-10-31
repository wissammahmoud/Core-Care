"""
LangGraph State Definition - Updated with LLM Analysis
======================================================

This state definition includes all fields needed for the complete nutrition analysis workflow,
including the new LLM analysis layer.

Replace your existing app/graphs/states.py with this file.
"""

from typing import TypedDict, Optional, Dict, List, Any
from PIL import Image


class NutritionState(TypedDict, total=False):
    """
    Complete state for the nutrition analysis workflow.
    
    Workflow stages:
    1. load_image: Sets 'image'
    2. inference: Sets 'raw_text'
    3. parse: Sets 'portions' and 'parsed_json'
    4. database_lookup: Sets 'milvus_results'
    5. llm_analysis: Sets 'llm_analysis'  ← NEW!
    6. calculate_nutrition: Sets 'nutrition_result'
    
    Fields:
        # Input
        image (Optional[Image.Image]): PIL Image object from user upload
        image_path (Optional[str]): Path to image file (alternative to image)
        instruction (str): User's query or instruction
        
        # Inference outputs
        raw_text (str): Raw JSON output from vision model
        parsed_json (Optional[Dict]): Parsed JSON from raw_text
        
        # Extracted data
        portions (Dict[str, float]): Ingredient names and their grams
            Example: {"chicken breast": 150, "brown rice": 200}
        
        # Database lookup
        milvus_results (List[Dict]): Top-k matches from Milvus for each ingredient
            Example: [
                {
                    "query": "chicken breast",
                    "results": [
                        {
                            "entity": {
                                "name": "chicken breast, grilled",
                                "calories": 165,
                                "protein": 31,
                                "carb": 0,
                                "fat": 3.6,
                                ...
                            },
                            "score": 0.95
                        },
                        ...
                    ]
                },
                ...
            ]
        
        # LLM Analysis (NEW!)
        llm_analysis (Optional[Dict[str, Any]]): Comprehensive nutritional analysis
            Example: {
                "dish_name": "Grilled Chicken with Brown Rice",
                "description": "A balanced meal...",
                "ingredient_selections": [
                    {
                        "detected_ingredient": "chicken breast",
                        "detected_grams": 150,
                        "selected_match": {
                            "name": "chicken breast, grilled",
                            "reason": "Matches cooking method...",
                            "macros_per_100g": {...},
                            "scaled_macros": {...}
                        }
                    },
                    ...
                ],
                "total_nutrition": {
                    "calories": 450,
                    "protein": 45,
                    "carbs": 50,
                    "fat": 8,
                    "fiber": 3
                },
                "healthiness": {
                    "score": 8.5,
                    "assessment": "...",
                    "pros": [...],
                    "cons": [...]
                },
                "micronutrients": {
                    "vitamins": [...],
                    "minerals": [...],
                    "other": [...]
                },
                "training_recovery": {
                    "pre_workout": "...",
                    "post_workout": "...",
                    "overall_impact": "..."
                },
                "satiety": {
                    "fullness_rating": 8,
                    "duration": "...",
                    "explanation": "..."
                },
                "quality_of_life": {
                    "short_term": "...",
                    "long_term": "...",
                    "frequency_recommendation": "..."
                }
            }
        
        # Nutrition calculation (from original calculate_nutrition node)
        nutrition_result (Optional[Dict]): Final nutrition summary
            Example: {
                "per_ingredient": [
                    {
                        "name": "chicken breast",
                        "grams": 150,
                        "macros": {...}
                    },
                    ...
                ],
                "totals": {
                    "grams": 350,
                    "calories": 450,
                    "protein": 45,
                    "carb": 50,
                    "fat": 8
                },
                "missing_ingredients": []
            }
        
        # Error handling
        error (Optional[str]): Error message if something went wrong
        error_stage (Optional[str]): Which stage the error occurred in
    """
    
    # Input fields
    image: Optional[Image.Image]
    image_path: Optional[str]
    instruction: str
    
    # Processing outputs
    raw_text: str
    parsed_json: Optional[Dict[str, Any]]
    portions: Dict[str, float]
    milvus_results: List[Dict[str, Any]]
    
    # LLM Analysis (NEW!)
    llm_analysis: Optional[Dict[str, Any]]
    
    # Final results
    nutrition_result: Optional[Dict[str, Any]]
    
    # Error handling
    error: Optional[str]
    error_stage: Optional[str]


# ============================================================================
# STATE INITIALIZATION HELPERS
# ============================================================================

def create_initial_state(
    image: Optional[Image.Image] = None,
    image_path: Optional[str] = None,
    instruction: str = "Analyze this food image and identify the ingredients with their portions."
) -> NutritionState:
    """
    Create an initial state for the workflow.
    
    Args:
        image: PIL Image object
        image_path: Path to image file
        instruction: User instruction
        
    Returns:
        Initialized NutritionState
    """
    return NutritionState(
        image=image,
        image_path=image_path,
        instruction=instruction,
        raw_text="",
        parsed_json=None,
        portions={},
        milvus_results=[],
        llm_analysis=None,
        nutrition_result=None,
        error=None,
        error_stage=None
    )


def is_state_valid(state: NutritionState) -> bool:
    """
    Check if a state has valid required fields.
    
    Args:
        state: State to validate
        
    Returns:
        True if state is valid
    """
    if state.get("error"):
        return False
    
    if not state.get("instruction"):
        return False
    
    # At least one input source required
    if not state.get("image") and not state.get("image_path"):
        return False
    
    return True


def get_state_summary(state: NutritionState) -> str:
    """
    Get a human-readable summary of the current state.
    
    Args:
        state: State to summarize
        
    Returns:
        Summary string
    """
    lines = []
    lines.append("="*60)
    lines.append("STATE SUMMARY")
    lines.append("="*60)
    
    # Input
    has_image = "✅" if state.get("image") else "❌"
    has_path = "✅" if state.get("image_path") else "❌"
    lines.append(f"Image: {has_image} | Path: {has_path}")
    lines.append(f"Instruction: {state.get('instruction', 'None')[:50]}...")
    
    # Processing
    has_raw = "✅" if state.get("raw_text") else "❌"
    has_parsed = "✅" if state.get("parsed_json") else "❌"
    lines.append(f"Raw text: {has_raw} | Parsed JSON: {has_parsed}")
    
    portions = state.get("portions", {})
    lines.append(f"Portions: {len(portions)} ingredients")
    if portions:
        lines.append(f"  → {', '.join(list(portions.keys())[:3])}")
    
    milvus = state.get("milvus_results", [])
    lines.append(f"Database results: {len(milvus)} matches")
    
    # LLM Analysis
    llm_analysis = state.get("llm_analysis")
    if llm_analysis:
        dish = llm_analysis.get("dish_name", "Unknown")
        cals = llm_analysis.get("total_nutrition", {}).get("calories", 0)
        health = llm_analysis.get("healthiness", {}).get("score", 0)
        lines.append(f"LLM Analysis: ✅")
        lines.append(f"  → Dish: {dish}")
        lines.append(f"  → Calories: {cals:.0f}")
        lines.append(f"  → Health Score: {health}/10")
    else:
        lines.append(f"LLM Analysis: ❌")
    
    # Final results
    nutrition = state.get("nutrition_result")
    if nutrition:
        totals = nutrition.get("totals", {})
        lines.append(f"Final nutrition: ✅ ({totals.get('calories', 0):.0f} cal)")
    else:
        lines.append(f"Final nutrition: ❌")
    
    # Errors
    if state.get("error"):
        lines.append(f"❌ ERROR: {state['error']}")
        if state.get("error_stage"):
            lines.append(f"   Stage: {state['error_stage']}")
    else:
        lines.append("✅ No errors")
    
    lines.append("="*60)
    return "\n".join(lines)