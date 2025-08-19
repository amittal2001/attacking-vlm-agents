"""
Utility functions for image processing and other tasks.
"""
import os
import logging
import torch
import wandb
from PIL import Image
from pgd import config

def load_image(image_path: str) -> Image.Image:
    """
    Loads an image from the given path.

    Args:
        image_path: The path to the image file.

    Returns:
        A PIL Image object.
    """
    image = Image.open(image_path).convert("RGB")
    return image

def save_image(tensor: torch.Tensor, file_path: str):
    """
    Saves a tensor as an image file.

    Args:
        tensor: The image tensor to save.
        file_path: The path where the image will be saved.
    """
    # Ensure the tensor is on the CPU and in the correct format
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    # If the tensor has a batch dimension, remove it
    if tensor.dim() == 4 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    
    # Convert to PIL Image
    # The tensor is expected to be in [C, H, W] format with values in [0, 1]
    image = Image.fromarray((tensor.permute(1, 2, 0).detach().numpy() * 255).astype('uint8'))
    
    # Save the image
    image.save(file_path)

def setup_logging(log_dir="logs"):
    """
    Sets up logging to both a file and the console.
    
    Args:
        log_dir (str): The directory to save log files in.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, "mip_generation.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    # Suppress overly verbose logs from third-party libraries.
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

def initialize_wandb():
    """
    Initializes a new Weights & Biases run.

    Returns:
        A W&B run object if initialization is successful, otherwise None.
    """
    if not config.WANDB_LOGGING:
        logging.info("W&B logging is disabled in the configuration.")
        return None
    try:
        run = wandb.init(
            project=config.WANDB_PROJECT,
            config={
                "model_id": config.MODEL_ID,
                "system_prompt": config.SYSTEM_PROMPT,
                "user_prompt": config.USER_PROMPT,
                "target_text": config.TARGET_TEXT,
                "eps": config.EPS,
                "alpha": config.ALPHA,
                "steps": config.STEPS,
            }
        )
        logging.info(f"W&B run initialized successfully. Run name: {run.name}")
        return run
    except Exception as e:
        logging.error(f"Failed to initialize W&B: {e}")
        return None
