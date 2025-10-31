"""
LangGraph Service - Main Workflow Runner
=========================================
Provides the main entry point for running the nutrition workflow.
"""
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def run_nutrition_workflow(
    image_bytes: Optional[bytes] = None,
    instruction: str = "Analyze the ingredients in this food.",
    system_prompt: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run the complete nutrition analysis workflow.
    
    Args:
        image_bytes: Optional image data (for vision analysis)
        instruction: Text instruction/query
        system_prompt: Optional system prompt override
    
    Returns:
        Dict containing the complete workflow state
    """
    from app.graphs.workflows import build_nutrition_workflow, get_workflow_summary
    
    try:
        logger.info("="*60)
        logger.info("🚀 Starting Nutrition Workflow")
        logger.info("="*60)
        
        # CRITICAL: Validate image_bytes type
        if image_bytes is not None:
            logger.info(f"Image bytes type: {type(image_bytes)}")
            logger.info(f"Image bytes length: {len(image_bytes) if isinstance(image_bytes, bytes) else 'N/A'}")
            
            # Check if it's actually bytes
            if not isinstance(image_bytes, bytes):
                error_msg = f"❌ CRITICAL ERROR: image_bytes must be bytes, got {type(image_bytes)}"
                logger.error(error_msg)
                logger.error(f"   Value preview: {str(image_bytes)[:200]}")
                return {
                    "error": error_msg,
                    "error_stage": "input_validation"
                }
            
            logger.info(f"✅ Image validated: {len(image_bytes)} bytes")
        else:
            logger.info("❌ No image (text-only mode)")
        
        logger.info(f"Instruction: {instruction[:100]}...")
        
        # Build workflow
        logger.info("🔧 Building workflow...")
        workflow = build_nutrition_workflow()
        logger.info("✅ Workflow compiled")
        
        # Prepare initial state - image_bytes MUST be bytes or None!
        initial_state = {
            "image": image_bytes,  # bytes or None, NOT dict!
            "instruction": instruction,
            "system_prompt": system_prompt
        }
        
        logger.info(f"Initial state prepared:")
        logger.info(f"  - image type: {type(image_bytes)}")
        logger.info(f"  - instruction: {instruction[:50]}...")
        logger.info(f"  - system_prompt: {system_prompt[:50] if system_prompt else 'None'}...")
        
        # Run workflow
        logger.info("🏃 Running workflow...")
        final_state = workflow.invoke(initial_state)
        logger.info("✅ Workflow completed")
        
        # Print summary
        summary = get_workflow_summary(final_state)
        logger.info("\n" + summary)
        
        # Check for errors
        if final_state.get("error"):
            logger.error(f"❌ Workflow error: {final_state['error']}")
            logger.error(f"   Stage: {final_state.get('error_stage', 'unknown')}")
        
        return final_state
        
    except Exception as e:
        logger.error(f"❌ Workflow execution failed: {e}", exc_info=True)
        return {
            "error": str(e),
            "error_stage": "workflow_execution"
        }


def format_workflow_response(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format the workflow state into a clean API response.
    
    Args:
        state: The final workflow state
    
    Returns:
        Formatted response dictionary
    """
    # Check for errors
    if state.get("error"):
        return {
            "success": False,
            "error": state["error"],
            "error_stage": state.get("error_stage", "unknown")
        }
    
    # Build success response
    response = {
        "success": True,
        "response": {},
        "mode": "image_analysis"
    }
    
    # Include vision analysis if available
    if state.get("parsed_json"):
        response["vision_analysis"] = state["parsed_json"]
    
    # Include database results if available
    if state.get("milvus_results"):
        response["database_matches"] = state["milvus_results"]
    
    # Include LLM analysis if available (this is the enhanced analysis)
    if state.get("llm_analysis"):
        response["response"] = state["llm_analysis"]
    else:
        # Fallback to basic analysis if LLM analysis not available
        portions = state.get("portions", {})
        response["resonse"] = {
            "ingredients": [
                {"name": name, "grams": grams}
                for name, grams in portions.items()
            ],
            "note": "Basic analysis only - LLM enhancement unavailable"
        }
    
    return response