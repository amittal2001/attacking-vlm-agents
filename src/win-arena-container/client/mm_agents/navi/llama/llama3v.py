
import base64, io, requests
from typing import Union, List, Optional
from PIL import Image
import torch
from transformers import AutoProcessor, MllamaForConditionalGeneration
import torchvision.transforms as T
import logging
import sys
from collections import Counter
from sentence_transformers import SentenceTransformer, util


# Configure logging to output to stdout
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] %(levelname)s %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger()
class Llama3Vision:
    def __init__(self, model_id: str = "meta-llama/Llama-3.2-11B-Vision-Instruct",
                 device: Optional[str] = None,
                 torch_dtype: Optional[torch.dtype] = None,
                 use_quantized: bool = False,
                 local_files_only: bool = False):
        self.model_id = model_id
        self.local_files_only = local_files_only
        # auto-select device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # recommended dtype on GPU
        if torch_dtype is None:
            self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        else:
            self.torch_dtype = torch_dtype

        # Load processor (handles image + text preproc)
        logger.info(f"Loading processor from {model_id} (local_files_only={local_files_only})")
        self.processor = AutoProcessor.from_pretrained(model_id, local_files_only=local_files_only)

        # Load model (use low_cpu_mem_usage and dtype hints)
        # If you plan to use a quantized checkpoint, set use_quantized=True and ensure repository supports that.
        logger.info(f"Loading model {model_id} on {self.device} dtype={self.torch_dtype}")
        # NOTE: when using large models on CPU you might need to set device_map / offload to disk (use accelerate or bitsandbytes)
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            local_files_only=local_files_only,
            # device_map="auto"  # uncomment if you want Transformers to auto-place layers (needs accelerate)
        )

        # move to device (if not using device_map)
        if getattr(self.model, "to", None) is not None and self.device != "cpu":
            self.model.to(self.device)

        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.sim_model = SentenceTransformer('all-MiniLM-L6-v2')

    # --- helpers to encode or wrap image ---
    def encode_image(self, image: Union[str, Image.Image], format="JPEG") -> str:
        if isinstance(image, str):
            with open(image, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        elif isinstance(image, Image.Image):
            image = image.convert("RGB")
            buf = io.BytesIO()
            image.save(buf, format=format)
            return base64.b64encode(buf.getvalue()).decode("utf-8")

    def get_url_payload(self, url: str) -> dict:
        return {"type": "image_url", "image_url": {"url": url}}

    def get_base64_payload(self, base64_image: str, format="JPEG") -> dict:
        mime = "jpeg" if format.upper() == "JPEG" else "png"
        return {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{base64_image}"}}

    # --- main multimodal call ---
    def pgd_process_images(self,
                       system_prompt: str,
                       question: str,
                       images: Union[torch.Tensor, Image.Image, List[Image.Image]],
                       targeted_plan_result: str,
                       max_tokens=512,
                       temperature=0.0,
                       only_text=True,
                       format="JPEG") -> tuple:
        # Always expect a single image
        if isinstance(images, list):
            if len(images) != 1:
                raise ValueError(f"Expected a single image, got {len(images)}.")
            images = images[0]
        # If tensor, convert to PIL Image
        if isinstance(images, torch.Tensor):
            if images.dim() == 4 and images.shape[0] == 1:
                images = images.squeeze(0)
            if images.dim() == 3:
                # Clamp and convert to uint8 if needed
                if images.max() <= 1.0:
                    images = images.clamp(0, 1)
                    images = (images * 255).to(torch.uint8)
                images = T.ToPILImage()(images.cpu())
            else:
                raise ValueError("Tensor must be shape [C,H,W] or [1,C,H,W]")
        if not isinstance(images, Image.Image):
            raise ValueError("images must be a PIL Image or list of one image.")

        prompt = (system_prompt or "You are a helpful assistant.") + "\n\n<|image|>\n" + question

        logger.info(f"Device: {self.device}")

        inputs = self.processor(images=images, text=prompt, return_tensors="pt")
        for k, v in inputs.items():
            if torch.is_floating_point(v):
                inputs[k] = v.to(self.device, dtype=self.torch_dtype)
            else:
                inputs[k] = v.to(self.device)

        inputs["pixel_values"].requires_grad_(requires_grad=True)
        gen_kwargs = dict(max_new_tokens=max_tokens,
                          do_sample=(temperature > 0),
                          temperature=temperature,
                          top_p=0.95)

        # Ensure aspect_ratio_ids is long if present
        if "aspect_ratio_ids" in inputs:
            inputs["aspect_ratio_ids"] = inputs["aspect_ratio_ids"].long()

        # Forward pass
        outputs = self.model(**inputs)
        logits = outputs.logits  # [batch, seq_len, vocab_size]

        pred_ids = logits.argmax(dim=-1)
        text = self.processor.decode(pred_ids[0], skip_special_tokens=True)
        # Compute loss between model output and targeted_plan_result
        # Tokenize the target string
        target_tokens = self.processor.tokenizer(
            targeted_plan_result,
            return_tensors="pt",
            add_special_tokens=False
        )["input_ids"].to(self.device)

        # Align target length with logits sequence length
        # logits: [batch, seq_len, vocab_size], target_tokens: [1, tgt_len]
        # We'll use the first batch (batch=0)
        seq_len = logits.shape[1]
        tgt_len = target_tokens.shape[1]
        min_len = min(seq_len, tgt_len)
        # Use only the overlapping part for loss
        logits_for_loss = logits[0, :min_len, :]
        targets_for_loss = target_tokens[0, :min_len]

        # CrossEntropyLoss expects input [N, C] and target [N] (class indices)
        loss = self.loss_fn(logits_for_loss, targets_for_loss)
        loss.backward()

        grad=inputs["pixel_values"].grad.mean(dim=2).squeeze().sign()
        
        return loss.item(), text, grad
    

    def process_images(self, system_prompt: str, question: str, images: Union[str, Image.Image, List[Union[str, Image.Image]]], max_tokens=30, temperature=0, only_text=True, format="JPEG") -> str:
        # Always expect a single image
        if isinstance(images, list):
            if len(images) != 1:
                raise ValueError(f"Expected a single image, got {len(images)}.")
            images = images[0]
        # If tensor, convert to PIL Image
        if isinstance(images, torch.Tensor):
            if images.dim() == 4 and images.shape[0] == 1:
                images = images.squeeze(0)
            if images.dim() == 3:
                # Clamp and convert to uint8 if needed
                if images.max() <= 1.0:
                    images = images.clamp(0, 1)
                    images = (images * 255).to(torch.uint8)
                images = T.ToPILImage()(images.cpu())
            else:
                raise ValueError("Tensor must be shape [C,H,W] or [1,C,H,W]")
        if not isinstance(images, Image.Image):
            raise ValueError("images must be a PIL Image or list of one image.")

        prompt = (system_prompt or "You are a helpful assistant.") + "\n\n<|image|>\n" + question

        logger.info(f"Device: {self.device}")

        inputs = self.processor(images=images, text=prompt, return_tensors="pt")
        for k, v in inputs.items():
            if torch.is_floating_point(v):
                inputs[k] = v.to(self.device, dtype=self.torch_dtype)
            else:
                inputs[k] = v.to(self.device)

        # Ensure aspect_ratio_ids is long if present
        if "aspect_ratio_ids" in inputs:
            inputs["aspect_ratio_ids"] = inputs["aspect_ratio_ids"].long()

        # Generate
        outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)

        if only_text:
            prompt_len = inputs["input_ids"].shape[1]
            return self.processor.decode(outputs[0][prompt_len:], skip_special_tokens=True)
        return outputs
    
    def choose_consistent_prediction(self, preds, threshold=0.8):
        """
        Returns the majority prediction only if it's semantically close
        (>= threshold) to at least half of the other predictions.
        Otherwise returns None.
        """
        if not preds:
            return None

        # Majority prediction
        majority_pred, _ = Counter(preds).most_common(1)[0]

        # Encode once
        emb_majority = self.sim_model.encode(majority_pred, convert_to_tensor=True)
        emb_all = self.sim_model.encode(preds, convert_to_tensor=True)

        # Compute cosine similarities
        sims = util.cos_sim(emb_majority, emb_all).squeeze().cpu().numpy()

        logger.info(f"sims: {sims}")
        logger.info(f"preds: {preds}")

        # Count how many are >= threshold (excluding self if you want)
        count_close = (sims >= threshold).sum()

        # Require at least half of all predictions to be close
        if count_close >= len(preds) / 2:
            return majority_pred
        else:
            return "no answer"
        
    def process_images_rand_smooth(self, system_prompt: str, question: str, images: Union[str, Image.Image, List[Union[str, Image.Image]]], max_tokens=30, temperature=0, only_text=True, format="JPEG", N=10, sigma=0.1) -> str:
        # Always expect a single image
        if isinstance(images, list):
            if len(images) != 1:
                raise ValueError(f"Expected a single image, got {len(images)}.")
            images = images[0]
        # If tensor, convert to PIL Image
        if isinstance(images, torch.Tensor):
            if images.dim() == 4 and images.shape[0] == 1:
                images = images.squeeze(0)
            if images.dim() == 3:
                # Clamp and convert to uint8 if needed
                if images.max() <= 1.0:
                    images = images.clamp(0, 1)
                    images = (images * 255).to(torch.uint8)
                images = T.ToPILImage()(images.cpu())
            else:
                raise ValueError("Tensor must be shape [C,H,W] or [1,C,H,W]")
        if not isinstance(images, Image.Image):
            raise ValueError("images must be a PIL Image or list of one image.")

        add_noise = T.Compose([
            T.ToTensor(),
            T.Lambda(lambda x: torch.clamp(x + torch.randn_like(x) * sigma, 0., 1.)),
            T.ToPILImage()
        ])

        preds = []
        for n in range(N):
          noisy_image = add_noise(images.copy())
          prompt = (system_prompt or "You are a helpful assistant.") + "\n\n<|image|>\n" + question
          inputs = self.processor(images=noisy_image, text=prompt, return_tensors="pt")
          for k, v in inputs.items():
              if torch.is_floating_point(v):
                  inputs[k] = v.to(self.device, dtype=self.torch_dtype)
              else:
                  inputs[k] = v.to(self.device)
          if "aspect_ratio_ids" in inputs:
              inputs["aspect_ratio_ids"] = inputs["aspect_ratio_ids"].long()
          # Generate
          outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)
          # Get how many tokens came from the prompt
          prompt_len = inputs["input_ids"].shape[1]
          text = self.processor.decode(outputs[0][prompt_len:], skip_special_tokens=True)
          logger.info(f"{n}'th answer: {text}")
          preds.append(text)
        smoothed_pred = self.choose_consistent_prediction(preds, threshold=0.8)
        return smoothed_pred
