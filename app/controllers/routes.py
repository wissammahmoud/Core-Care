"""
Flask Routes - Fixed Image Bytes Handling
=========================================
Properly handles image upload and passes raw bytes to workflow.
"""
from flask import request, jsonify, Blueprint
from werkzeug.utils import secure_filename
import logging

logger = logging.getLogger(__name__)

# Create blueprint
api_bp = Blueprint('nutrition', __name__)


def allowed_file(filename):
    """Check if file extension is allowed."""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@api_bp.route('/api/analyze', methods=['POST'])
def analyze():
    """
    Analyze food from image or text description.
    
    Accepts:
    - multipart/form-data with 'image' file and optional 'instruction'
    - application/json with 'instruction' (text-only)
    
    Returns:
    - JSON with nutrition analysis
    """
    try:
        logger.info("="*60)
        logger.info("📥 New analysis request")
        
        # Initialize variables
        image_bytes = None
        instruction = None
        
        # ===============================================================
        # CASE 1: Multipart form data (could be image OR text-only)
        # ===============================================================
        if request.files or request.form:
            logger.info("📦 Request type: Multipart form data")
            
            # Try to get instruction first
            instruction = request.form.get('instruction')
            
            # Check for image in either 'image' or 'file' field
            file = None
            if 'image' in request.files:
                file = request.files['image']
                logger.info("Found 'image' field in request")
            elif 'file' in request.files:
                file = request.files['file']
                logger.info("Found 'file' field in request")
            
            # Handle image if provided
            if file and file.filename != '':
                logger.info(f"📸 Processing image: {file.filename}")
                
                # Check file extension
                if not allowed_file(file.filename):
                    logger.error(f"Invalid file type: {file.filename}")
                    return jsonify({"error": "Invalid file type. Allowed: png, jpg, jpeg, gif, webp, bmp"}), 400
                
                # Read image bytes - CRITICAL: Read as raw bytes!
                image_bytes = file.read()
                
                # Verify it's actually bytes
                if not isinstance(image_bytes, bytes):
                    logger.error(f"❌ Image is not bytes! Type: {type(image_bytes)}")
                    return jsonify({"error": "Failed to read image as bytes"}), 500
                
                logger.info(f"✅ Image loaded: {len(image_bytes)} bytes")
                logger.info(f"✅ Image type verified: {type(image_bytes)}")
                
                # Set default instruction if not provided
                if not instruction:
                    instruction = 'Analyze the ingredients in this food image.'
            
            else:
                # No image file or empty filename - TEXT-ONLY mode
                logger.info("📝 No image provided - TEXT-ONLY mode")
                
                if not instruction:
                    logger.error("No instruction provided for text-only mode")
                    return jsonify({
                        "error": "No image or instruction provided. Please provide either an image file or an instruction."
                    }), 400
                
                logger.info("✅ Text-only request validated")
            
            logger.info(f"Instruction: {instruction}")
        
        # ===============================================================
        # CASE 2: JSON data (text-only)
        # ===============================================================
        elif request.is_json:
            logger.info("📦 Request type: JSON (text-only)")
            data = request.get_json()
            instruction = data.get('instruction')
            
            if not instruction:
                logger.error("No 'instruction' in JSON data")
                return jsonify({"error": "Missing 'instruction' field"}), 400
            
            logger.info(f"Instruction: {instruction}")
        
        # ===============================================================
        # CASE 3: Invalid request
        # ===============================================================
        else:
            logger.error("Invalid request format")
            return jsonify({
                "error": "Invalid request format. Use multipart/form-data or application/json"
            }), 400
        
        # ===============================================================
        # Run workflow based on request type
        # ===============================================================
        
        # CASE A: Text-only - Use base LLM directly (no nutrition workflow)
        if not image_bytes:
            logger.info("="*60)
            logger.info("🤖 TEXT-ONLY MODE: Using base LLM directly")
            logger.info("="*60)
            logger.info(f"Question: {instruction}")
            
            # Get base model (not LoRa)
            from app.services.model_adapter import get_model_and_processor
            
            try:
                logger.info("Loading base model for text generation...")
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
                        "content": instruction
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
                logger.info("🚀 Generating response...")
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
                
                logger.info(f"✅ Generated response: {len(response_text)} characters")
                logger.info("="*60)
                
                return jsonify({
                    "status": "success",
                    "response": response_text,
                    "mode": "text_only"
                })
                
            except Exception as e:
                logger.error(f"❌ Text generation failed: {e}", exc_info=True)
                return jsonify({
                    "status": "error",
                    "error": str(e),
                    "mode": "text_only"
                }), 500
        
        # CASE B: Image provided - Use nutrition workflow
        logger.info("="*60)
        logger.info("📸 IMAGE MODE: Using nutrition workflow")
        logger.info("="*60)
        logger.info("🚀 Starting workflow...")
        logger.info(f"  Image: {'✅ ' + str(len(image_bytes)) + ' bytes' if image_bytes else '❌ None'}")
        logger.info(f"  Image type: {type(image_bytes)}")
        logger.info(f"  Instruction: {instruction[:50]}...")
        
        from app.services.langgraph_service import run_nutrition_workflow, format_workflow_response
        
        # CRITICAL: Pass image_bytes directly as bytes, NOT in a dict!
        final_state = run_nutrition_workflow(
            image_bytes=image_bytes,  # This should be bytes or None, NOT a dict!
            instruction=instruction
        )
        
        # Format response
        response = format_workflow_response(final_state)
        
        logger.info("✅ Workflow completed")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"❌ Route handler error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@api_bp.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok', 'service': 'nutrition-analysis'})


# ============================================================================
# Registration function for apps that don't use blueprints
# ============================================================================

def register_routes(app):
    """
    Register routes with Flask app (without blueprint).
    
    Use this if your app doesn't use blueprints:
        from app.controllers.routes import register_routes
        register_routes(app)
    """
    
    @app.route('/api/analyze', methods=['POST'])
    def analyze():
        """
        Analyze food from image or text description.
        """
        try:
            logger.info("="*60)
            logger.info("📥 New analysis request")
            
            # Initialize variables
            image_bytes = None
            instruction = None
            
            # Check for multipart form data
            if request.files:
                logger.info("📦 Request type: Multipart (image upload)")
                
                # Check for image in either 'image' or 'file' field
                file = None
                if 'image' in request.files:
                    file = request.files['image']
                    logger.info("Found image in 'image' field")
                elif 'file' in request.files:
                    file = request.files['file']
                    logger.info("Found image in 'file' field")
                else:
                    logger.error("No 'image' or 'file' field")
                    return jsonify({"error": "No image file provided"}), 400
                
                if file.filename == '':
                    return jsonify({"error": "Empty filename"}), 400
                
                if not allowed_file(file.filename):
                    return jsonify({"error": "Invalid file type"}), 400
                
                # Read as raw bytes
                logger.info(f"Processing: {file.filename}")
                image_bytes = file.read()
                
                # Verify type
                if not isinstance(image_bytes, bytes):
                    logger.error(f"❌ Not bytes! Type: {type(image_bytes)}")
                    return jsonify({"error": "Failed to read image"}), 500
                
                logger.info(f"✅ Image: {len(image_bytes)} bytes")
                logger.info(f"✅ Type: {type(image_bytes)}")
                
                instruction = request.form.get('instruction', 'Analyze the ingredients in this food image.')
            
            # Check for JSON data
            elif request.is_json:
                logger.info("📦 Request type: JSON (text-only)")
                data = request.get_json()
                instruction = data.get('instruction')
                
                if not instruction:
                    return jsonify({"error": "Missing 'instruction'"}), 400
            
            else:
                return jsonify({"error": "Invalid request format"}), 400
            
            # Run workflow
            logger.info("🚀 Starting workflow...")
            logger.info(f"  Image: {type(image_bytes)} - {'✅ ' + str(len(image_bytes)) + ' bytes' if image_bytes else '❌ None'}")
            
            from app.services.langgraph_service import run_nutrition_workflow, format_workflow_response
            
            # Pass image_bytes directly
            final_state = run_nutrition_workflow(
                image_bytes=image_bytes,
                instruction=instruction
            )
            
            response = format_workflow_response(final_state)
            
            logger.info("✅ Workflow completed")
            return jsonify(response)
            
        except Exception as e:
            logger.error(f"❌ Error: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/health', methods=['GET'])
    def health():
        """Health check."""
        return jsonify({'status': 'ok', 'service': 'nutrition-analysis'})
    
    logger.info("✅ Routes registered")

# """
# Flask Routes - Fixed Image Bytes Handling
# =========================================
# Properly handles image upload and passes raw bytes to workflow.
# """
# from flask import request, jsonify, Blueprint
# from werkzeug.utils import secure_filename
# import logging

# logger = logging.getLogger(__name__)

# # Create blueprint
# api_bp = Blueprint('nutrition', __name__)


# def allowed_file(filename):
#     """Check if file extension is allowed."""
#     ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp'}
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# @api_bp.route('/api/analyze', methods=['POST'])
# def analyze():
#     """
#     Analyze food from image or text description.
    
#     Accepts:
#     - multipart/form-data with 'image' file and optional 'instruction'
#     - application/json with 'instruction' (text-only)
    
#     Returns:
#     - JSON with nutrition analysis
#     """
#     try:
#         logger.info("="*60)
#         logger.info("📥 New analysis request")
        
#         # Initialize variables
#         image_bytes = None
#         instruction = None
        
#         # ===============================================================
#         # CASE 1: Multipart form data (image upload)
#         # ===============================================================
#         if request.files:
#             logger.info("📦 Request type: Multipart (image upload)")
            
#             # Check for image in either 'image' or 'file' field
#             file = None
#             if 'image' in request.files:
#                 file = request.files['image']
#                 logger.info("Found image in 'image' field")
#             elif 'file' in request.files:
#                 file = request.files['file']
#                 logger.info("Found image in 'file' field")
#             else:
#                 logger.error("No 'image' or 'file' field in form data")
#                 return jsonify({"error": "No image file provided. Use 'image' or 'file' field."}), 400
            
#             # Check if file was actually selected
#             if file.filename == '':
#                 logger.error("Empty filename")
#                 return jsonify({"error": "Empty filename"}), 400
            
#             # Check file extension
#             if not allowed_file(file.filename):
#                 logger.error(f"Invalid file type: {file.filename}")
#                 return jsonify({"error": "Invalid file type. Allowed: png, jpg, jpeg, gif, webp, bmp"}), 400
            
#             # Read image bytes - CRITICAL: Read as raw bytes!
#             logger.info(f"Processing image: {file.filename}")
#             image_bytes = file.read()
            
#             # Verify it's actually bytes
#             if not isinstance(image_bytes, bytes):
#                 logger.error(f"❌ Image is not bytes! Type: {type(image_bytes)}")
#                 return jsonify({"error": "Failed to read image as bytes"}), 500
            
#             logger.info(f"✅ Image loaded: {len(image_bytes)} bytes")
#             logger.info(f"✅ Image type verified: {type(image_bytes)}")
            
#             # Get instruction from form (optional)
#             instruction = request.form.get('instruction', 'Analyze the ingredients in this food image.')
#             logger.info(f"Instruction: {instruction}")
        
#         # ===============================================================
#         # CASE 2: JSON data (text-only)
#         # ===============================================================
#         elif request.is_json:
#             logger.info("📦 Request type: JSON (text-only)")
#             data = request.get_json()
#             instruction = data.get('instruction')
            
#             if not instruction:
#                 logger.error("No 'instruction' in JSON data")
#                 return jsonify({"error": "Missing 'instruction' field"}), 400
            
#             logger.info(f"Instruction: {instruction}")
        
#         # ===============================================================
#         # CASE 3: Invalid request
#         # ===============================================================
#         else:
#             logger.error("Invalid request format")
#             return jsonify({
#                 "error": "Invalid request format. Use multipart/form-data or application/json"
#             }), 400
        
#         # ===============================================================
#         # Run workflow with PROPER image_bytes (not wrapped in dict!)
#         # ===============================================================
#         logger.info("🚀 Starting workflow...")
#         logger.info(f"  Image: {'✅ ' + str(len(image_bytes)) + ' bytes' if image_bytes else '❌ None'}")
#         logger.info(f"  Image type: {type(image_bytes)}")
#         logger.info(f"  Instruction: {instruction[:50]}...")
        
#         from app.services.langgraph_service import run_nutrition_workflow, format_workflow_response
        
#         # CRITICAL: Pass image_bytes directly as bytes, NOT in a dict!
#         final_state = run_nutrition_workflow(
#             image_bytes=image_bytes,  # This should be bytes or None, NOT a dict!
#             instruction=instruction
#         )
        
#         # Format response
#         response = format_workflow_response(final_state)
        
#         logger.info("✅ Workflow completed")
#         return jsonify(response)
        
#     except Exception as e:
#         logger.error(f"❌ Route handler error: {e}", exc_info=True)
#         return jsonify({"error": str(e)}), 500


# @api_bp.route('/api/health', methods=['GET'])
# def health():
#     """Health check endpoint."""
#     return jsonify({'status': 'ok', 'service': 'nutrition-analysis'})


# # ============================================================================
# # Registration function for apps that don't use blueprints
# # ============================================================================

# def register_routes(app):
#     """
#     Register routes with Flask app (without blueprint).
    
#     Use this if your app doesn't use blueprints:
#         from app.controllers.routes import register_routes
#         register_routes(app)
#     """
    
#     @app.route('/api/analyze', methods=['POST'])
#     def analyze():
#         """
#         Analyze food from image or text description.
#         """
#         try:
#             logger.info("="*60)
#             logger.info("📥 New analysis request")
            
#             # Initialize variables
#             image_bytes = None
#             instruction = None
            
#             # Check for multipart form data
#             if request.files:
#                 logger.info("📦 Request type: Multipart (image upload)")
                
#                 # Check for image in either 'image' or 'file' field
#                 file = None
#                 if 'image' in request.files:
#                     file = request.files['image']
#                     logger.info("Found image in 'image' field")
#                 elif 'file' in request.files:
#                     file = request.files['file']
#                     logger.info("Found image in 'file' field")
#                 else:
#                     logger.error("No 'image' or 'file' field")
#                     return jsonify({"error": "No image file provided"}), 400
                
#                 if file.filename == '':
#                     return jsonify({"error": "Empty filename"}), 400
                
#                 if not allowed_file(file.filename):
#                     return jsonify({"error": "Invalid file type"}), 400
                
#                 # Read as raw bytes
#                 logger.info(f"Processing: {file.filename}")
#                 image_bytes = file.read()
                
#                 # Verify type
#                 if not isinstance(image_bytes, bytes):
#                     logger.error(f"❌ Not bytes! Type: {type(image_bytes)}")
#                     return jsonify({"error": "Failed to read image"}), 500
                
#                 logger.info(f"✅ Image: {len(image_bytes)} bytes")
#                 logger.info(f"✅ Type: {type(image_bytes)}")
                
#                 instruction = request.form.get('instruction', 'Analyze the ingredients in this food image.')
            
#             # Check for JSON data
#             elif request.is_json:
#                 logger.info("📦 Request type: JSON (text-only)")
#                 data = request.get_json()
#                 instruction = data.get('instruction')
                
#                 if not instruction:
#                     return jsonify({"error": "Missing 'instruction'"}), 400
            
#             else:
#                 return jsonify({"error": "Invalid request format"}), 400
            
#             # Run workflow
#             logger.info("🚀 Starting workflow...")
#             logger.info(f"  Image: {type(image_bytes)} - {'✅ ' + str(len(image_bytes)) + ' bytes' if image_bytes else '❌ None'}")
            
#             from app.services.langgraph_service import run_nutrition_workflow, format_workflow_response
            
#             # Pass image_bytes directly
#             final_state = run_nutrition_workflow(
#                 image_bytes=image_bytes,
#                 instruction=instruction
#             )
            
#             response = format_workflow_response(final_state)
            
#             logger.info("✅ Workflow completed")
#             return jsonify(response)
            
#         except Exception as e:
#             logger.error(f"❌ Error: {e}", exc_info=True)
#             return jsonify({"error": str(e)}), 500
    
#     @app.route('/api/health', methods=['GET'])
#     def health():
#         """Health check."""
#         return jsonify({'status': 'ok', 'service': 'nutrition-analysis'})
    
#     logger.info("✅ Routes registered")