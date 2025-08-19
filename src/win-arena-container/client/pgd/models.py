"""
Handles the loading of Vision-Language Models from Hugging Face.
"""
import torch
import logging
from transformers import AutoProcessor, AutoModelForCausalLM

# Get a logger for this module.
logger = logging.getLogger(__name__)

def load_model(model_id: str):
    """
    Loads a Vision-Language Model and its processor from the Hugging Face Hub.

    This function uses the AutoModelForCausalLM and AutoProcessor classes to
    load a wide range of models, making it flexible and easy to adapt.

    Args:
        model_id (str): The identifier of the model to load (e.g., "meta-llama/Llama-3.2-11B-Vision-Model").

    Returns:
        A tuple containing:
        - model (AutoModelForCausalLM): The loaded model.
        - processor (AutoProcessor): The corresponding processor.
    """
    logger.info(f"Starting model loading for model_id: {model_id}")
    
    # --- Load Model ---
    # AutoModelForCausalLM provides a generic interface for causal language models,
    # including vision-capable models like Llama 3.2 Vision.
    logger.info("Loading model from pretrained using AutoModelForCausalLM...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,  # Use float16 for memory efficiency.
            low_cpu_mem_usage=True,     # Optimizes memory usage on the CPU.
            trust_remote_code=True,     # Required for some models with custom code.
        )
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    # --- Load Processor ---
    # The processor handles both text tokenization and image preprocessing.
    logger.info("Loading processor from pretrained...")
    try:
        processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
        logger.info("Processor loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load processor: {e}")
        raise

    # --- Move to GPU if available ---
    if torch.cuda.is_available():
        logger.info("CUDA is available. Moving model to GPU...")
        try:
            model.to("cuda")
            logger.info("Model successfully moved to GPU.")
        except Exception as e:
            logger.error(f"Failed to move model to GPU: {e}")
            raise
    else:
        logger.warning("CUDA not available. Model will run on CPU.")

    return model, processor