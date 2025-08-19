"""
Implements the PGD-based white-box attack for Vision-Language Models.
"""
import torch
import torch.nn as nn
import logging
from tqdm import tqdm

# Get a logger for this module.
logger = logging.getLogger(__name__)

class VLMWhiteBoxPGDAttack:
    """
    Performs a Projected Gradient Descent (PGD) L-infinity white-box attack
    on a Vision-Language Model.

    This is a targeted attack that attempts to force the model to generate a
    specific target text sequence when presented with a perturbed image.
    """

    def __init__(self, model, processor, eps=8/255., n=50, alpha=1e-2,
                 rand_init=True, early_stop=True, wandb_run=None):
        """
        Initializes the PGD attack.

        Args:
            model: The VLM to be attacked.
            processor: The processor for the VLM, which includes the tokenizer.
            eps (float): The maximum perturbation allowed (L-infinity norm).
            n (int): The number of PGD steps (iterations).
            alpha (float): The learning rate or step size for each iteration.
            rand_init (bool): If True, starts with a random perturbation.
            early_stop (bool): If True, stops early if the target is achieved.
            wandb_run: An optional Weights & Biases run object for logging.
        """
        self.model = model
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.n = n
        self.alpha = alpha
        self.eps = eps
        self.rand_init = rand_init
        self.early_stop = early_stop
        self.wandb_run = wandb_run
        self.loss_func = nn.CrossEntropyLoss(reduction='mean')

        logger.info("VLMWhiteBoxPGDAttack initialized.")
        logger.info(f"  - Epsilon (eps): {self.eps:.4f}")
        logger.info(f"  - Alpha (learning rate): {self.alpha:.4f}")
        logger.info(f"  - Steps (n): {self.n}")
        logger.info(f"  - Random Init: {self.rand_init}")
        logger.info(f"  - Early Stopping: {self.early_stop}")

    def execute(self, pixel_values, input_ids_for_loss, attention_mask_for_loss,
                labels, image_sizes):
        """
        Executes the PGD attack on the input image.

        Args:
            pixel_values (torch.Tensor): The input image tensor.
            input_ids_for_loss (torch.Tensor): Token IDs for loss calculation (prompt + target).
            attention_mask_for_loss (torch.Tensor): Attention mask for the loss calculation.
            labels (torch.Tensor): The target output tokens, with prompt tokens masked.
            image_sizes (torch.Tensor): The original sizes of the images.

        Returns:
            torch.Tensor: The adversarially perturbed image tensor.
        """
        logger.info("Starting PGD attack execution.")
        # Clone the original image tensor to create the adversarial example.
        x_adv = pixel_values.clone().detach()

        # Start with a random perturbation if enabled.
        if self.rand_init:
            logger.info("Applying random initialization to the image tensor.")
            x_adv += torch.empty_like(pixel_values).uniform_(-self.eps, self.eps)
            x_adv = torch.clamp(x_adv, 0, 1) # Ensure the perturbation is valid.

        # The adversarial image requires gradients for optimization.
        x_adv.requires_grad = True

        # Use the Adam optimizer for more stable gradient updates.
        optimizer = torch.optim.Adam([x_adv], lr=self.alpha)

        target_mask = labels != -100
        target_indices = target_mask.nonzero(as_tuple=False)

        # PGD main loop
        for i in tqdm(range(self.n), desc="PGD Attack Steps"):
            optimizer.zero_grad()

            partial_labels = labels.clone()
            partial_labels[:] = -100

            if i < len(target_indices):
                unmask_up_to = i + 1
            else:
                unmask_up_to = len(target_indices)

            partial_indices = target_indices[:unmask_up_to]
            for index_pair in partial_indices:
                partial_labels[index_pair[0], index_pair[1]] = labels[index_pair[0], index_pair[1]]

            # Forward pass through the model to get the loss.
            outputs = self.model(
                pixel_values=x_adv,
                input_ids=input_ids_for_loss,
                attention_mask=attention_mask_for_loss,
                #labels=labels,
                labels=partial_labels,
                image_sizes=image_sizes,
                return_dict=True
            )
            loss = outputs.loss

            if i % 10 == 0:
                logger.info(f"Step [{i}/{self.n}] - Loss: {loss.item():.4f}")
            if self.wandb_run:
                self.wandb_run.log({"step": i, "loss": loss.item()})

            # Backward pass to compute gradients.
            loss.backward()

            # Zero out gradients for the dummy images in the stack.
            if x_adv.grad is not None and x_adv.shape[1] > 1:
                x_adv.grad.data[:, 1:, ...] = 0

            # Update the adversarial image.
            optimizer.step()

            # Project the perturbation back into the epsilon ball.
            with torch.no_grad():
                delta = torch.clamp(x_adv - pixel_values, -self.eps, self.eps)
                x_adv.data = pixel_values + delta
                x_adv.data = torch.clamp(x_adv.data, 0, 1) # Ensure valid image range.
            
            # Re-enable gradients for the next iteration.
            x_adv.requires_grad = True

            # --- Efficient Early Stopping Check ---
            if self.early_stop and (i % 10 == 0 or i == self.n - 1):
                with torch.no_grad():
                    # Reuse the logits from the forward pass to check probabilities.
                    logits = outputs.logits
                    target_token_indices = (labels != -100).nonzero(as_tuple=True)
                    relevant_logits = logits[target_token_indices]
                    target_token_ids = labels[target_token_indices]
                    
                    # If all target tokens have a high probability, stop early.
                    probs = torch.softmax(relevant_logits, dim=-1)
                    correct_token_probs = probs.gather(1, target_token_ids.unsqueeze(1)).squeeze()
                    if torch.all(correct_token_probs > 0.99):
                        logger.info(f"Early stopping at step {i}. All target token probabilities > 99%.")
                        if self.wandb_run:
                            self.wandb_run.log({"early_stop_step": i})
                        break
        
        logger.info("PGD attack finished.")
        return x_adv