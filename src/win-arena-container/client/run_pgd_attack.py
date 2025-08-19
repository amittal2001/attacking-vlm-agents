
"""
Main script to generate a Malicious Image Patch (MIP).

This script orchestrates the entire process of loading a Vision-Language Model,
applying a PGD-based adversarial attack to an input image, and saving the
resulting malicious image.
"""

import os
import torch
import wandb
import logging
from PIL import Image
from pgd import config
from pgd.models import load_model
from pgd.utils import load_image, save_image, setup_logging, initialize_wandb
from pgd.PGDAttacks import VLMWhiteBoxPGDAttack

def prepare_attack_inputs(processor, image, device):
    """
    Prepares the inputs required for the PGD attack and verification.

    This function formats the prompts using the model's chat template and
    tokenizes them, preparing tensors for the loss calculation, early stopping
    checks, and final verification.

    Args:
        processor: The model's processor for tokenization.
        image (PIL.Image): The input image.
        device (torch.device): The device to move tensors to (e.g., 'cuda').

    Returns:
        A dictionary containing all the prepared tensors.
    """
    logging.info("Preparing inputs for the attack using chat template...")

    # --- Prepare inputs for the early stopping check (prompt only) ---
    prompt_only_messages = [
        {"role": "system", "content": config.SYSTEM_PROMPT},
        {"role": "user", "content": f"<image>\n{config.USER_PROMPT}"},
    ]
    formatted_prompt_only = processor.tokenizer.apply_chat_template(
        prompt_only_messages, tokenize=False, add_generation_prompt=True
    )
    inputs_for_gen = processor(
        text=formatted_prompt_only, images=image, return_tensors="pt"
    ).to(device)

    # --- Prepare inputs for the loss calculation (includes target text) ---
    target_messages = [
        {"role": "system", "content": config.SYSTEM_PROMPT},
        {"role": "user", "content": f"<image>\n{config.USER_PROMPT}"},
        {"role": "assistant", "content": config.TARGET_TEXT}
    ]
    formatted_target_prompt = processor.tokenizer.apply_chat_template(
        target_messages, tokenize=False, add_generation_prompt=False
    )
    inputs_for_loss = processor(
        text=formatted_target_prompt, images=image, return_tensors="pt"
    ).to(device)

    # --- Create labels for the loss function ---
    # The labels are the input_ids of the target, with the prompt part masked.
    labels = inputs_for_loss['input_ids'].clone()
    prompt_length = inputs_for_gen['input_ids'].shape[1]
    labels[:, :prompt_length] = -100  # Mask out the prompt tokens

    logging.info("Inputs for loss calculation and early stopping have been prepared.")

    return {
        "pixel_values": inputs_for_loss['pixel_values'],
        "input_ids_for_loss": inputs_for_loss['input_ids'],
        "attention_mask_for_loss": inputs_for_loss['attention_mask'],
        "input_ids_for_gen": inputs_for_gen['input_ids'],
        "attention_mask_for_gen": inputs_for_gen['attention_mask'],
        "labels": labels,
        "image_sizes": inputs_for_loss.get('image_sizes'),
    }

def verify_attack(model, processor, image_path, device):
    """
    Verifies the effectiveness of the attack by generating text from the
    adversarial image and checking if it matches the target text.

    Args:
        model: The VLM to use for verification.
        processor: The model's processor.
        image_path (str): The path to the saved adversarial image.
        device (torch.device): The device to run the model on.

    Returns:
        The generated text from the model.
    """
    logging.info("Verifying the attack effectiveness...")
    try:
        adv_image = Image.open(image_path)
        logging.info("Successfully loaded saved adversarial image for verification.")
    except Exception as e:
        logging.error(f"Could not load the saved image for verification: {e}")
        return None

    # Format the prompt for verification using the chat template.
    verify_messages = [
        {"role": "system", "content": config.SYSTEM_PROMPT},
        {"role": "user", "content": f"<image>\n{config.USER_PROMPT}"}
    ]
    formatted_verify_prompt = processor.tokenizer.apply_chat_template(
        verify_messages, tokenize=False, add_generation_prompt=True
    )

    # Process the text and adversarial image.
    verify_inputs = processor(
        text=formatted_verify_prompt, images=adv_image, return_tensors="pt"
    ).to(device)

    # Manually get the input embeddings by passing the text and image data
    # through the main body of the model. This bypasses the `generate`
    # function's strict input validation while still preparing the correct
    # inputs for the language model head.
    with torch.no_grad():
        # The model's main body (`model.model`) returns the last hidden state,
        # which serves as the input embeddings for the LM head.
        inputs_embeds = model.model(
            input_ids=verify_inputs["input_ids"],
            pixel_values=verify_inputs["pixel_values"],
            attention_mask=verify_inputs["attention_mask"],
        )[0]

    # Now, call generate with the embeddings.
    generation_kwargs = {
        "inputs_embeds": inputs_embeds,
        "attention_mask": verify_inputs["attention_mask"],
    }

    with torch.no_grad():
        output = model.generate(**generation_kwargs, max_new_tokens=100)

    # Since we started from embeddings, the output contains the full sequence.
    # We decode the entire output and extract the assistant's response.
    generated_text = processor.decode(output[0], skip_special_tokens=True)
    
    try:
        # Isolate the assistant's reply by finding the end of the user prompt.
        assistant_response_start = generated_text.rfind(config.USER_PROMPT) + len(config.USER_PROMPT)
        verification_result = generated_text[assistant_response_start:].strip()
    except Exception:
        # Fallback in case the prompt isn't found in the output.
        verification_result = generated_text.split("ASSISTANT:")[-1].strip()

    logging.info(f"Verification - Model's output: '{verification_result}'")
    return verification_result


def main():
    """
    Main function to run the MIP generation pipeline.
    """
    setup_logging()
    logging.info("--- Starting MIP Generation ---")

    # --- 0. Initialize W&B ---
    wandb_run = initialize_wandb()

    # --- 1. Setup Paths ---
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(config.OUTPUT_DIR, config.OUTPUT_IMAGE_NAME)
    logging.info(f"Output path set to: {output_path}")

    # --- 2. Load Model and Processor ---
    logging.info(f"Loading model and processor for: {config.MODEL_ID}")
    model, processor = load_model(config.MODEL_ID)
    device = next(model.parameters()).device
    logging.info(f"Model loaded on device: {device}")

    # --- 3. Load Image ---
    logging.info(f"Loading input image from: {config.INPUT_IMAGE_PATH}")
    image = load_image(config.INPUT_IMAGE_PATH)
    if wandb_run:
        wandb_run.log({"original_image": wandb.Image(image)})

    # --- 4. Prepare Inputs for the Attack ---
    attack_inputs = prepare_attack_inputs(processor, image, device)

    # --- 5. Initialize and Run Attack ---
    logging.info("Initializing PGD attack...")
    attack = VLMWhiteBoxPGDAttack(
        model,
        processor,
        eps=config.EPS,
        n=config.STEPS,
        alpha=config.ALPHA,
        wandb_run=wandb_run
    )
    
    logging.info("Starting PGD attack execution...")
    adversarial_image_tensor = attack.execute(
        pixel_values=attack_inputs["pixel_values"],
        input_ids_for_loss=attack_inputs["input_ids_for_loss"],
        attention_mask_for_loss=attack_inputs["attention_mask_for_loss"],
        labels=attack_inputs["labels"],
        image_sizes=attack_inputs["image_sizes"],
    )
    logging.info("PGD attack finished.")

    # --- 6. Save the Adversarial Image ---
    # The tensor from the attack can have extra dimensions (e.g., for batch size or dummy images).
    # We squeeze out dimensions of size 1 and then select the first image from any remaining stack
    # to ensure a 3D [C, H, W] tensor is passed to the save function.
    image_to_save = adversarial_image_tensor.squeeze()
    if image_to_save.dim() > 3:
        image_to_save = image_to_save[0]

    logging.info(f"Saving adversarial image to: {output_path}")
    save_image(image_to_save, output_path)
    if wandb_run:
        wandb_run.log({"adversarial_image": wandb.Image(Image.open(output_path))})

    # --- 7. Verification ---
    verification_result = verify_attack(model, processor, output_path, device)
    if wandb_run and verification_result:
        wandb_run.log({"verification_output": verification_result})
    
    if wandb_run:
        wandb_run.finish()
        logging.info("W&B run finished.")
    
    logging.info("--- MIP Generation Complete ---")

if __name__ == "__main__":
    # Adjust the current working directory if running from the project root.
    if os.path.basename(os.getcwd()) == 'attacking_agents':
        os.chdir('mip_generator')
    main()