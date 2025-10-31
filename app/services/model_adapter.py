"""
Model Adapter - Auto-Initializing Version
==========================================
Automatically initializes models with correct parameters.
"""
import logging
import torch

logger = logging.getLogger(__name__)


def get_model_and_processor(has_image: bool = True):
    """
    Get model and processor from ModelManager.
    Automatically initializes if needed.
    
    Args:
        has_image: Whether the task involves images (True) or text-only (False)
    
    Returns:
        tuple: (model, processor)
    """
    from app.services.model_service import ModelManager
    
    try:
        logger.info(f"Getting model and processor (has_image={has_image})...")
        
        # Get ModelManager instance
        model_mgr = ModelManager.get_instance()
        logger.info(f"✅ Got ModelManager instance")
        
        # =================================================================
        # Try to get model - if it fails, auto-initialize
        # =================================================================
        result = None
        model = None
        processor = None
        
        try:
            result = model_mgr.get_model_for_input(has_image=has_image)
            logger.info(f"Result type: {type(result)}")
        except Exception as e:
            logger.warning(f"⚠️ get_model_for_input() failed: {e}")
            result = None
        
        # =================================================================
        # If result is None or tuple with all None, try to initialize
        # =================================================================
        needs_init = False
        
        if result is None:
            needs_init = True
            logger.warning("⚠️ get_model_for_input() returned None")
        elif isinstance(result, tuple):
            # Check if all elements are None
            if all(x is None for x in result):
                needs_init = True
                logger.warning("⚠️ All tuple elements are None")
        
        if needs_init:
            logger.info("🔧 AUTO-INITIALIZING MODELS...")
            
            # Try to find model name
            model_name = find_model_name(model_mgr)
            
            if model_name is None:
                # Use default
                model_name = "Qwen/Qwen2-VL-7B-Instruct"
                logger.warning(f"⚠️ Using default model: {model_name}")
            else:
                logger.info(f"✅ Found model name: {model_name}")
            
            # Check initialize_models signature
            import inspect
            sig = inspect.signature(model_mgr.initialize_models)
            params = list(sig.parameters.keys())
            
            logger.info(f"initialize_models parameters: {params}")
            
            # Call with correct parameters
            if 'base_model_id' in params:
                logger.info(f"Calling initialize_models(base_model_id='{model_name}')")
                model_mgr.initialize_models(base_model_id=model_name)
            else:
                # Set base_model_id attribute first
                if hasattr(model_mgr, 'base_model_id'):
                    model_mgr.base_model_id = model_name
                    logger.info(f"Set model_mgr.base_model_id = '{model_name}'")
                logger.info("Calling initialize_models()")
                model_mgr.initialize_models()
            
            logger.info("✅ Initialization complete, trying again...")
            
            # Try getting model again
            result = model_mgr.get_model_for_input(has_image=has_image)
        
        # =================================================================
        # Parse result
        # =================================================================
        if isinstance(result, tuple):
            logger.info(f"✅ Got tuple with {len(result)} elements")
            
            # Log contents
            for i, item in enumerate(result):
                if item is None:
                    logger.info(f"  [{i}] = None")
                else:
                    logger.info(f"  [{i}] = {type(item).__name__}")
            
            # Smart identification
            logger.info("🔍 Identifying model and processor...")
            
            # Find model (has 'parameters' method)
            for i, item in enumerate(result):
                if item is not None and hasattr(item, 'parameters'):
                    try:
                        next(item.parameters(), None)
                        model = item
                        logger.info(f"✅ Found MODEL at [{i}]: {type(item).__name__}")
                        break
                    except:
                        pass
            
            # Find processor (has 'Processor' or 'Tokenizer' in name)
            for i, item in enumerate(result):
                if item is not None:
                    name = type(item).__name__
                    if 'Processor' in name or 'Tokenizer' in name:
                        processor = item
                        logger.info(f"✅ Found PROCESSOR at [{i}]: {name}")
                        break
                    elif hasattr(item, 'apply_chat_template'):
                        processor = item
                        logger.info(f"✅ Found PROCESSOR at [{i}]: {name} (has apply_chat_template)")
                        break
            
            # Fallback: use non-None items in order
            if model is None or processor is None:
                non_none = [(i, x) for i, x in enumerate(result) if x is not None]
                if len(non_none) >= 2:
                    if model is None:
                        model = non_none[0][1]
                        logger.warning(f"⚠️ Using [{non_none[0][0]}] as model: {type(model).__name__}")
                    if processor is None:
                        processor = non_none[1][1]
                        logger.warning(f"⚠️ Using [{non_none[1][0]}] as processor: {type(processor).__name__}")
        
        elif result is not None:
            model = result
            logger.info(f"✅ Got model: {type(model).__name__}")
        
        # =================================================================
        # Verify and get processor if missing
        # =================================================================
        if model is None:
            raise ValueError("Could not find model after initialization!")
        
        if processor is None:
            logger.warning("⚠️ No processor found, trying alternatives...")
            
            # Try ModelManager attributes
            if hasattr(model_mgr, 'processor') and model_mgr.processor:
                processor = model_mgr.processor
                logger.info(f"✅ Got from model_mgr.processor")
            elif hasattr(model_mgr, 'tokenizer') and model_mgr.tokenizer:
                processor = model_mgr.tokenizer
                logger.info(f"✅ Got from model_mgr.tokenizer")
            elif hasattr(model, 'processor') and model.processor:
                processor = model.processor
                logger.info(f"✅ Got from model.processor")
            else:
                # Auto-load processor
                model_name = find_model_name(model_mgr) or find_model_name_from_model(model)
                if model_name:
                    processor = load_processor(model_name, model_mgr)
                else:
                    raise ValueError("Could not determine model name to load processor")
        
        # =================================================================
        # Setup device
        # =================================================================
        device = get_model_device(model)
        if not hasattr(model, 'device'):
            try:
                model.device = device
            except:
                pass
        
        logger.info("="*60)
        logger.info("✅ SUCCESS! Model and processor ready!")
        logger.info(f"   Model: {type(model).__name__}")
        logger.info(f"   Processor: {type(processor).__name__}")
        logger.info(f"   Device: {device}")
        logger.info("="*60)
        
        return model, processor
        
    except Exception as e:
        logger.error(f"❌ Failed: {e}", exc_info=True)
        raise


def find_model_name(model_mgr):
    """Try to find model name from ModelManager."""
    # Check base_model_id
    if hasattr(model_mgr, 'base_model_id') and model_mgr.base_model_id:
        return model_mgr.base_model_id
    
    # Check env var
    import os
    if 'MODEL_NAME' in os.environ:
        return os.environ['MODEL_NAME']
    
    # Check HuggingFace cache
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    if os.path.exists(cache_dir):
        dirs = os.listdir(cache_dir)
        qwen_dirs = [d for d in dirs if 'qwen' in d.lower() and 'models--' in d]
        if qwen_dirs:
            dir_name = qwen_dirs[0]
            if dir_name.startswith('models--'):
                parts = dir_name.replace('models--', '').split('--')
                if len(parts) >= 2:
                    return '/'.join(parts)
    
    return None


def find_model_name_from_model(model):
    """Try to find model name from model object."""
    if hasattr(model, 'config') and hasattr(model.config, '_name_or_path'):
        return model.config._name_or_path
    if hasattr(model, 'name_or_path'):
        return model.name_or_path
    return None


def load_processor(model_name, model_mgr):
    """Load processor from HuggingFace."""
    try:
        from transformers import AutoProcessor
        logger.info(f"Loading processor: {model_name}")
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        
        # Cache it
        if hasattr(model_mgr, 'processor'):
            model_mgr.processor = processor
        
        logger.info(f"✅ Loaded: {type(processor).__name__}")
        return processor
    except Exception as e:
        logger.error(f"Failed to load processor: {e}")
        raise


def get_model_device(model):
    """Get model device."""
    if hasattr(model, 'device'):
        return model.device
    try:
        return next(model.parameters()).device
    except:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')