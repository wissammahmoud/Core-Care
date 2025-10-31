"""
Enhanced Model service with LoRA adapter support.
Handles:
- Base model for text-only inputs
- Base model + LoRA adapter for image inputs
"""
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from threading import Lock
import logging
import os
import sys

logger = logging.getLogger(__name__)


class ModelManager:
    """Singleton model manager with LoRA support for image inputs"""
    _instance = None
    _lock = Lock()
    
    def __init__(self):
        self.base_model = None
        self.lora_model = None
        self.processor = None
        self.tokenizer = None
        self.device = None
        self.is_quantized = False
        self.lora_loaded = False
        self.base_model_id = os.getenv('MODEL_ID', 'unsloth/Qwen3-VL-4B-Instruct-bnb-4bit')
        self.lora_adapter_id = os.getenv('LORA_ADAPTER_ID', 'WissMah/Qwen2.5VL-FT-Lora_mix_aug1')
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def load_base_model(self, model_id: str):
        """Load base model (without LoRA)"""
        if self.base_model is not None and self.base_model_id == model_id:
            return self.base_model, self.processor, self.tokenizer
        
        with self._lock:
            if self.base_model is not None and self.base_model_id == model_id:
                return self.base_model, self.processor, self.tokenizer
            
            logger.info(f"Loading base model: {model_id}")
            self.base_model_id = model_id
            
            # Disable torch compilation
            os.environ['TORCH_COMPILE_DISABLE'] = '1'
            torch._dynamo.config.suppress_errors = True
            torch._dynamo.config.disable = True
            
            # Determine device and loading strategy
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                cuda_version = torch.version.cuda
                gpu_name = torch.cuda.get_device_name(0)
                total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
                
                logger.info(f"GPU detected: {gpu_name}")
                logger.info(f"CUDA version: {cuda_version}")
                logger.info(f"Total VRAM: {total_vram:.2f} GB")
                torch.cuda.empty_cache()
                
                use_quantization = True
            else:
                self.device = torch.device("cpu")
                logger.warning("GPU not available, using CPU (will be slow)")
                use_quantization = False
            
            # Determine attention implementation
            attn_impl = "sdpa" if sys.platform.startswith("win") else "flash_attention_2"
            
            # Load base model
            try:
                if use_quantization:
                    logger.info("Loading base model with 4-bit quantization (GPU)")
                    
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                    
                    self.base_model = AutoModelForImageTextToText.from_pretrained(
                        model_id,
                        quantization_config=quantization_config,
                        device_map="auto",
                        attn_implementation=attn_impl,
                        low_cpu_mem_usage=True,
                        trust_remote_code=True,
                    )
                    self.is_quantized = True
                    
                else:
                    logger.info("Loading base model without quantization (CPU)")
                    
                    try:
                        self.base_model = AutoModelForImageTextToText.from_pretrained(
                            model_id,
                            torch_dtype=torch.float16,
                            device_map="cpu",
                            low_cpu_mem_usage=True,
                            trust_remote_code=True,
                        )
                    except Exception:
                        logger.warning("Float16 failed, trying float32")
                        self.base_model = AutoModelForImageTextToText.from_pretrained(
                            model_id,
                            torch_dtype=torch.float32,
                            device_map="cpu",
                            low_cpu_mem_usage=True,
                            trust_remote_code=True,
                        )
                    
                    self.is_quantized = False
                
                # Load processor and tokenizer
                self.processor = AutoProcessor.from_pretrained(
                    model_id, 
                    trust_remote_code=True
                )
                self.tokenizer = self.processor.tokenizer
                
                # Set model to eval mode
                self.base_model.eval()
                
                device_info = next(self.base_model.parameters()).device
                logger.info(f"Base model loaded successfully on: {device_info}")
                
                return self.base_model, self.processor, self.tokenizer
                
            except Exception as e:
                logger.error(f"Failed to load base model: {e}")
                raise

    def load_lora_adapter(self, lora_adapter_id: str):
        """Load LoRA adapter on top of base model"""
        if self.base_model is None:
            raise ValueError("Base model must be loaded before loading LoRA adapter")
        
        if self.lora_loaded and self.lora_adapter_id == lora_adapter_id:
            logger.info("LoRA adapter already loaded")
            return self.lora_model, self.processor, self.tokenizer
        
        with self._lock:
            logger.info(f"Loading LoRA adapter: {lora_adapter_id}")
            self.lora_adapter_id = lora_adapter_id
            
            try:
                # Load LoRA adapter using PEFT
                self.lora_model = PeftModel.from_pretrained(
                    self.base_model,
                    lora_adapter_id,
                    is_trainable=False,
                )
                
                self.lora_model.eval()
                self.lora_loaded = True
                
                logger.info("LoRA adapter loaded successfully")
                logger.info("Using LoRA model for image inputs")
                
                return self.lora_model, self.processor, self.tokenizer
                
            except Exception as e:
                logger.error(f"Failed to load LoRA adapter: {e}")
                logger.warning("Falling back to base model only")
                self.lora_loaded = False
                return self.base_model, self.processor, self.tokenizer
    
    def get_model_for_input(self, has_image: bool):
        """
        Get appropriate model based on input type.
        
        Args:
            has_image: True if input contains image, False for text-only
        
        Returns:
            tuple: (model, processor, tokenizer)
        """
        if has_image and self.lora_loaded:
            logger.info("Using LoRA model (image input)")
            return self.lora_model, self.processor, self.tokenizer
        else:
            logger.info("Using base model (text-only input)")
            return self.base_model, self.processor, self.tokenizer
    
    def initialize_models(self, base_model_id: str, lora_adapter_id: str = None):
        """
        Initialize both base model and optionally LoRA adapter.
        
        Args:
            base_model_id: HuggingFace model ID for base model
            lora_adapter_id: HuggingFace model ID for LoRA adapter (optional)
        """
        # Load base model
        self.load_base_model(base_model_id)
        
        # Load LoRA adapter if provided
        if lora_adapter_id:
            try:
                self.load_lora_adapter(lora_adapter_id)
            except Exception as e:
                logger.warning(f"Could not load LoRA adapter: {e}")
                logger.info("Continuing with base model only")
    
    def cleanup(self):
        """Clean up GPU memory"""
        if torch.cuda.is_available() and self.device and self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def unload_lora(self):
        """Unload LoRA adapter, keep base model"""
        with self._lock:
            if self.lora_model is not None:
                del self.lora_model
                self.lora_model = None
                self.lora_loaded = False
                self.lora_adapter_id = None
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                logger.info("LoRA adapter unloaded")
    
    def unload_all(self):
        """Unload everything"""
        with self._lock:
            if self.lora_model is not None:
                del self.lora_model
            if self.base_model is not None:
                del self.base_model
            if self.processor is not None:
                del self.processor
            if self.tokenizer is not None:
                del self.tokenizer
            
            self.base_model = None
            self.lora_model = None
            self.processor = None
            self.tokenizer = None
            self.lora_loaded = False
            self.base_model_id = None
            self.lora_adapter_id = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("All models unloaded")
    
    def get_model_info(self):
        """Get information about loaded models"""
        info = {
            "base_model_loaded": self.base_model is not None,
            "base_model_id": self.base_model_id,
            "lora_loaded": self.lora_loaded,
            "lora_adapter_id": self.lora_adapter_id,
            "device": str(self.device) if self.device else None,
            "quantized": self.is_quantized,
        }
        
        if torch.cuda.is_available() and self.device and self.device.type == "cuda":
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["allocated_memory_gb"] = torch.cuda.memory_allocated(0) / 1024**3
            info["cached_memory_gb"] = torch.cuda.memory_reserved(0) / 1024**3
        
        return info