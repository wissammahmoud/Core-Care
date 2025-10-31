"""
Detailed ModelManager Diagnostic
=================================
Run this to see EXACTLY what's in your ModelManager.
"""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

def detailed_diagnostic():
    print("="*70)
    print("🔍 DETAILED ModelManager DIAGNOSTIC")
    print("="*70)
    
    try:
        from app.services.model_service import ModelManager
        print("✅ ModelManager imported")
    except Exception as e:
        print(f"❌ Failed to import: {e}")
        return
    
    try:
        mgr = ModelManager.get_instance()
        print("✅ Got instance\n")
    except Exception as e:
        print(f"❌ Failed to get instance: {e}")
        return
    
    # Check every attribute
    print("="*70)
    print("📊 ATTRIBUTE VALUES")
    print("="*70)
    
    attrs_to_check = [
        'base_model', 'lora_model', 'processor', 'tokenizer',
        'base_model_id', 'lora_adapter_id', 'device', 
        'is_quantized', 'lora_loaded'
    ]
    
    for attr in attrs_to_check:
        if hasattr(mgr, attr):
            value = getattr(mgr, attr)
            if value is None:
                print(f"  {attr:<20} = None")
            elif isinstance(value, str):
                print(f"  {attr:<20} = '{value}'")
            elif isinstance(value, bool):
                print(f"  {attr:<20} = {value}")
            else:
                print(f"  {attr:<20} = <{type(value).__name__}>")
        else:
            print(f"  {attr:<20} = [NOT FOUND]")
    
    print("\n" + "="*70)
    print("🔍 CHECKING base_model_id")
    print("="*70)
    
    if hasattr(mgr, 'base_model_id'):
        base_model_id = mgr.base_model_id
        print(f"  Value: {base_model_id}")
        print(f"  Type: {type(base_model_id)}")
        print(f"  Is None: {base_model_id is None}")
        print(f"  Is empty: {base_model_id == '' if base_model_id else 'N/A'}")
        
        if base_model_id:
            print(f"  ✅ base_model_id is set: '{base_model_id}'")
        else:
            print(f"  ❌ base_model_id is None or empty!")
    else:
        print("  ❌ base_model_id attribute doesn't exist!")
    
    print("\n" + "="*70)
    print("🔍 TESTING get_model_for_input()")
    print("="*70)
    
    if hasattr(mgr, 'get_model_for_input'):
        try:
            model = mgr.get_model_for_input(has_image=True)
            if model is None:
                print("  ❌ Returns None")
            else:
                print(f"  ✅ Returns: {type(model).__name__}")
                
                # Check if model has config
                if hasattr(model, 'config'):
                    print(f"  Model has config: True")
                    if hasattr(model.config, '_name_or_path'):
                        print(f"  Model name from config: {model.config._name_or_path}")
                    else:
                        print(f"  Model config has no _name_or_path")
                else:
                    print(f"  Model has no config attribute")
        except Exception as e:
            print(f"  ❌ Error: {e}")
    else:
        print("  ❌ Method doesn't exist")
    
    print("\n" + "="*70)
    print("🔍 CHECKING PROCESSOR")
    print("="*70)
    
    # Check processor
    if hasattr(mgr, 'processor'):
        proc = mgr.processor
        if proc is None:
            print("  ❌ processor = None")
        else:
            print(f"  ✅ processor = {type(proc).__name__}")
    else:
        print("  ❌ processor attribute doesn't exist")
    
    # Check tokenizer
    if hasattr(mgr, 'tokenizer'):
        tok = mgr.tokenizer
        if tok is None:
            print("  ❌ tokenizer = None")
        else:
            print(f"  ✅ tokenizer = {type(tok).__name__}")
    else:
        print("  ❌ tokenizer attribute doesn't exist")
    
    print("\n" + "="*70)
    print("🔍 METHODS AVAILABLE")
    print("="*70)
    
    methods = ['initialize_models', 'load_base_model', 'load_lora_adapter', 
               'get_model_for_input', 'get_model_info']
    
    for method in methods:
        if hasattr(mgr, method):
            print(f"  ✅ {method}()")
        else:
            print(f"  ❌ {method}() [NOT FOUND]")
    
    print("\n" + "="*70)
    print("🔧 TRYING TO LOAD PROCESSOR MANUALLY")
    print("="*70)
    
    # Try to load processor manually
    if hasattr(mgr, 'base_model_id') and mgr.base_model_id:
        try:
            from transformers import AutoProcessor
            print(f"  Attempting: AutoProcessor.from_pretrained('{mgr.base_model_id}')")
            processor = AutoProcessor.from_pretrained(mgr.base_model_id)
            print(f"  ✅ SUCCESS! Loaded: {type(processor).__name__}")
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
    else:
        print("  ❌ Cannot try - base_model_id is not set")
        
        # Try from model config
        if hasattr(mgr, 'get_model_for_input'):
            try:
                model = mgr.get_model_for_input(has_image=True)
                if model and hasattr(model, 'config') and hasattr(model.config, '_name_or_path'):
                    model_name = model.config._name_or_path
                    print(f"  Found model name from config: '{model_name}'")
                    
                    from transformers import AutoProcessor
                    print(f"  Attempting: AutoProcessor.from_pretrained('{model_name}')")
                    processor = AutoProcessor.from_pretrained(model_name)
                    print(f"  ✅ SUCCESS! Loaded: {type(processor).__name__}")
            except Exception as e:
                print(f"  ❌ FAILED: {e}")
    
    print("\n" + "="*70)
    print("✅ DIAGNOSTIC COMPLETE ")
    print("="*70)


if __name__ == "__main__":
    detailed_diagnostic()