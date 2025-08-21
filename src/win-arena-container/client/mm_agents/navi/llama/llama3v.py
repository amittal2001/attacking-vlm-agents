import base64, io, requests
from typing import Union, List, Optional
from PIL import Image
import torch
from transformers import AutoProcessor, MllamaForConditionalGeneration

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
                       images: Union[str, Image.Image, List[Union[str, Image.Image]]],
                       max_tokens=512,
                       temperature=0.0,
                       only_text=True,
                       format="JPEG") -> str:

        if not isinstance(images, list):
            images = [images]

        # For HF llama, we don’t actually send `content` JSON; we need PIL images.
        # So convert any base64/url payloads into a real PIL.Image
        pil_images = []
        for img in images:
            if isinstance(img, str) and img.startswith("http"):
                # download
                resp = requests.get(img)
                pil_images.append(Image.open(io.BytesIO(resp.content)).convert("RGB"))
            elif isinstance(img, str):  # local path
                pil_images.append(Image.open(img).convert("RGB"))
            elif isinstance(img, Image.Image):
                pil_images.append(img.convert("RGB"))
            else:
                raise ValueError(f"Unsupported image input type: {type(img)}")

        # Build text prompt
        prompt = (system_prompt or "You are a helpful assistant.") + "\n\n" + question

        # Currently llama-3.2 vision supports single image best
        if len(pil_images) > 1:
            print("⚠️ Warning: multi-image may not be fully supported, using first image.")
        image = pil_images[0]

        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device, self.torch_dtype)

        gen_kwargs = dict(max_new_tokens=max_tokens,
                          do_sample=(temperature > 0),
                          temperature=temperature,
                          top_p=0.95)

        with torch.no_grad():
            output = self.model.generate(**inputs, **gen_kwargs)

        text = self.processor.decode(output[0], skip_special_tokens=True)
        return text