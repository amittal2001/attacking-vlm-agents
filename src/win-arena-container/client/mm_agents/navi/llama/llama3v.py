import base64, io, requests
from typing import Union, List, Optional
from PIL import Image
import torch
from transformers import AutoProcessor, MllamaForConditionalGeneration
import torchvision.transforms as T

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
            
            
            
        # convert images to tensor
        if isinstance(images, list):
            images = images [0]
        if isinstance(images, Image.Image):
            images = T.ToTensor()(images)  # -> [C,H,W]
            images = images.unsqueeze(0)   # -> [1,C,H,W]    
            # Ensure batch dimension
            
        if images.dim() == 3:   # [C,H,W]
            images = images.unsqueeze(0)  # -> [1,C,H,W]
        elif images.dim() != 4:
            raise ValueError(f"Unsupported tensor shape {images.shape}, expected [C,H,W] or [B,C,H,W]")

        # Build text prompt
        prompt = (system_prompt or "You are a helpful assistant.") + "\n\n<image>\n" + question

        inputs = self.processor(images=images, text=prompt, return_tensors="pt").to(self.device, self.torch_dtype)

        gen_kwargs = dict(max_new_tokens=max_tokens,
                          do_sample=(temperature > 0),
                          temperature=temperature,
                          top_p=0.95)

        output = self.model.generate(**inputs, **gen_kwargs)

        text = self.processor.decode(output[0], skip_special_tokens=True)
        return text