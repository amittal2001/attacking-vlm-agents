import base64, io, requests
from typing import Union, List, Optional
from PIL import Image
import torch
from transformers import AutoProcessor, MllamaForConditionalGeneration
import torchvision.transforms as T
import logging

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
        print(f"Loading processor from {model_id} (local_files_only={local_files_only})")
        self.processor = AutoProcessor.from_pretrained(model_id, local_files_only=local_files_only)

        # Load model (use low_cpu_mem_usage and dtype hints)
        # If you plan to use a quantized checkpoint, set use_quantized=True and ensure repository supports that.
        print(f"Loading model {model_id} on {self.device} dtype={self.torch_dtype}")
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
    def process_images(self,
                       system_prompt: str,
                       question: str,
                       images: Union[torch.Tensor, Image.Image, List[Image.Image]],
                       max_tokens=512,
                       temperature=0.0,
                       only_text=True,
                       format="JPEG") -> str:
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

        logger = logging.getLogger()
        logger.info(f"Processing images with prompt: {prompt}")
        logger.info(f"image type: {type(images)}")
        logger.info(f"image size: {images.size()}")

        prompt = (system_prompt or "You are a helpful assistant.") + "\n\n<|image|>\n" + question

        inputs = self.processor(images=images, text=prompt, return_tensors="pt").to(self.device, self.torch_dtype)
        inputs["pixel_values"].requires_grad_()
        gen_kwargs = dict(max_new_tokens=max_tokens,
                          do_sample=(temperature > 0),
                          temperature=temperature,
                          top_p=0.95)
        
        # --- Forward pass (not .generate) ---
        inputs = {k: v.to(self.device, self.torch_dtype) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        logits = outputs.logits  # [batch, seq_len, vocab_size]

        pred_ids = logits.argmax(dim=-1)
        text = self.processor.decode(pred_ids[0], skip_special_tokens=True)
        return text