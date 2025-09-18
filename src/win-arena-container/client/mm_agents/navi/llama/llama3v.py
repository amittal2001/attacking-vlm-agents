
from urllib import response
import base64, io, requests
from typing import Union, List, Optional
from PIL import Image
import torch
from transformers import AutoProcessor, MllamaForConditionalGeneration, BitsAndBytesConfig
import torchvision.transforms as T
import logging
import sys
from collections import Counter
from sentence_transformers import SentenceTransformer, util
import wandb
import gc


# Configure logging to output to stdout
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger()
class Llama3Vision:
    def __init__(self, model_id: str = "meta-llama/Llama-3.2-11B-Vision-Instruct",
                 device: Optional[str] = None,
                 dtype: Optional[torch.dtype] = None,
                 use_quantized: bool = False,
                 local_files_only: bool = False,):
        self.model_id = model_id
        self.local_files_only = local_files_only
        # auto-select device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # recommended dtype on GPU
        if dtype is None:
            self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        else:
            self.dtype = dtype

        # Load processor (handles image + text preproc)
        logger.info(f"Loading processor from {model_id} (local_files_only={local_files_only})")
        self.processor = AutoProcessor.from_pretrained(model_id, local_files_only=local_files_only)

        # Load model with 4-bit quantization using bitsandbytes
        logger.info(f"Loading model {model_id} in 4-bit quantized mode on {self.device} dtype={self.dtype}")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            dtype=self.dtype,
            low_cpu_mem_usage=True,
            local_files_only=local_files_only,
            #device_map="auto",
            quantization_config=quant_config
        )
        for p in self.model.parameters():
            p.requires_grad_(False)

        # move to device (if not using device_map)
        self.model.to(self.device)

        logger.info(f"Loading model all-MiniLM-L6-v2 on {self.device} dtype={self.dtype}")
    
        self.loss_fn = torch.nn.CrossEntropyLoss()

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
                           format="JPEG",
                           num_steps=1,
                           alpha=0.01,
                           epsilon=0.03,
                           early_stopping="True",
                           wandb_run=None) -> tuple:
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

        
        if wandb_run is not None:
            wandb_run.log({"original_image": wandb.Image(images)})
            prompt_table = wandb.Table(columns=["prompt"])
            prompt_table.add_data(prompt)
            wandb_run.log({"prompt": prompt_table})

        device = self.device
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Device: {device}")

        transform = T.ToTensor()
        image_tensor = transform(images.resize((560, 560)).convert("RGB")).unsqueeze(0).cpu()
        adv_image_tensor = image_tensor.clone().detach().cpu()
        best_loss = float("inf")
        best_adv_image = adv_image_tensor.clone().detach().cpu()

        if wandb_run is not None:
            columns = ["step", "response"]
            data = []

        for i in range(num_steps):
            adv_image = T.ToPILImage()(adv_image_tensor.clone().detach().squeeze().cpu())
            inputs = self.processor(images=adv_image, text=prompt, return_tensors="pt")

            for k, v in inputs.items():
                if torch.is_tensor(v):
                    inputs[k] = v.to(device)

            inputs["pixel_values"].requires_grad_(requires_grad=True)
            gen_kwargs = dict(max_new_tokens=max_tokens,
                            do_sample=(temperature > 0),
                            temperature=temperature,
                            top_p=0.95)

            # Ensure aspect_ratio_ids is long if present
            if "aspect_ratio_ids" in inputs:
                inputs["aspect_ratio_ids"] = inputs["aspect_ratio_ids"].long()

            # Forward pass
            logger.info(f"Starting {i} forward pass...")
            outputs = self.model(**inputs)
            logger.info(f"Finished {i} forward pass...")
            logits = outputs.logits  # [batch, seq_len, vocab_size]

            pred_ids = logits.argmax(dim=-1)
            next_token_id = pred_ids[0, -1].unsqueeze(0)
            text = self.processor.decode(next_token_id, skip_special_tokens=True)
            logger.info(f"Predicted text for step {i}: {text}")
            
            # Tokenize the target string
            target_tokens = self.processor.tokenizer(
                targeted_plan_result,
                return_tensors="pt",
                add_special_tokens=False
            )["input_ids"]

            next_token_logits = logits[0, -1].to(device)
            target_token = target_tokens[0, 0].to(device)
            loss = self.loss_fn(next_token_logits.unsqueeze(0), target_token.unsqueeze(0))

            logger.info(f"Loss for step {i}: {loss.item()}")
            logger.info(f"next token logits for step {i}: {next_token_logits.max().item()} for token {text}")
            logger.info(f"target token logits for step {i}: {next_token_logits[target_token.item()].item()} for token {targeted_plan_result}")

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_adv_image = adv_image_tensor.clone().detach().cpu()

            if early_stopping == "True" and text == targeted_plan_result:
                data.append([i, text])
                break

            grad = torch.autograd.grad(loss, inputs["pixel_values"], retain_graph=False, create_graph=False)[0]
            grad = grad[:, :, 0].squeeze().sign().cpu()

            with torch.no_grad():
                adv_image_tensor.data = adv_image_tensor.data - float(alpha) * grad
                adv_image_tensor.data = torch.min(torch.max(adv_image_tensor.data, image_tensor.data - float(epsilon)), image_tensor.data + float(epsilon))
                adv_image_tensor.data = torch.clamp(adv_image_tensor.data, 0, 1)  

            if wandb_run is not None:
                wandb_run.log({
                    "step": i,
                    "adv_image": wandb.Image(T.ToPILImage()(adv_image_tensor.clone().detach().squeeze().cpu())),
                    "loss": loss.item(),
                    "logits_max_next_token": next_token_logits.max().item(),
                    "logits_target_token": next_token_logits[target_token.item()].item()
                })
                data.append([i, text])

            for k, v in inputs.items():
                if torch.is_tensor(v):
                    del v
            del loss, grad, outputs, logits
            gc.collect()
            torch.cuda.empty_cache()

        if wandb_run is not None:
            response_table = wandb.Table(columns=columns, data=data)
            wandb.log({"model_responses": response_table})
            wandb_run.log({"final_adv_image": wandb.Image(T.ToPILImage()(best_adv_image.clone().detach().squeeze().cpu()))})
        return text, best_adv_image.detach().cpu()


    def process_images(self,
                       system_prompt: str,
                       question: str,
                       images: Union[str, Image.Image, List[Union[str, Image.Image]]],
                       max_tokens=100,
                       temperature=0,
                       only_text=True,
                       format="JPEG",
                       wandb_run=None) -> str:
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

        if wandb_run is not None:
            wandb_run.log({"input_image": wandb.Image(images)})
            prompt_table = wandb.Table(columns=["prompt"])
            prompt_table.add_data(prompt)
            wandb_run.log({"prompt": prompt_table})

        device = self.device
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Device: {device}")

        inputs = self.processor(images=images, text=prompt, return_tensors="pt")

        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(device)

        # Ensure aspect_ratio_ids is long if present
        if "aspect_ratio_ids" in inputs:
            inputs["aspect_ratio_ids"] = inputs["aspect_ratio_ids"].long()

        # Generate
        logger.info(f"Starting forward pass...")
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)
        logger.info(f"Finished forward pass...")

        prompt_len = inputs["input_ids"].shape[1]
        text = self.processor.decode(outputs[0][prompt_len:], skip_special_tokens=True)
        logger.info(f"Predicted text: {text}")

        if wandb_run is not None:
            table = wandb.Table(columns=["model_response"])
            table.add_data(text)
            wandb_run.log({"model_response": table})

        cpu_outputs = outputs.clone().detach().cpu()

        del inputs, outputs
        gc.collect()
        torch.cuda.empty_cache()

        if only_text:
            return text
        return cpu_outputs
    
    def choose_consistent_prediction(self, preds, threshold=0.8, sim_model=None):
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
        emb_majority = sim_model.encode(majority_pred, convert_to_tensor=True)
        emb_all = sim_model.encode(preds, convert_to_tensor=True)

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
        
    def process_images_rand_smooth(self,
                                   system_prompt: str,
                                   question: str,
                                   images: Union[str, Image.Image, List[Union[str, Image.Image]]],
                                   max_tokens=100,
                                   temperature=0,
                                   only_text=True,
                                   format="JPEG",
                                   N=10,
                                   sigma=0.1,
                                   wandb_run=None) -> str:
        sim_model = SentenceTransformer('all-MiniLM-L6-v2').to(self.device)
        for p in sim_model.parameters():
            p.requires_grad_(False)

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

        if wandb_run is not None:
            wandb_run.log({"input_image": wandb.Image(images)})
            prompt_table = wandb.Table(columns=["prompt"])
            prompt_table.add_data(prompt)
            wandb_run.log({"prompt": prompt_table})

        device = self.device
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Device: {device}")

        add_noise = T.Compose([
            T.ToTensor(),
            T.Lambda(lambda x: torch.clamp(x + torch.randn_like(x) * sigma, 0., 1.)),
            T.ToPILImage()
        ])

        preds = []
        if wandb_run is not None:
            columns = ["n", "response"]
            data = []
        
        for n in range(N):
            noisy_image = add_noise(images.copy())
            inputs = self.processor(images=noisy_image, text=prompt, return_tensors="pt")

            for k, v in inputs.items():
                if torch.is_tensor(v):
                    inputs[k] = v.to(device)

            if "aspect_ratio_ids" in inputs:
                inputs["aspect_ratio_ids"] = inputs["aspect_ratio_ids"].long()

            # Generate
            logger.info(f"Starting {n} forward pass...")
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)
            logger.info(f"Finished {n} forward pass...")

            # Get how many tokens came from the prompt
            prompt_len = inputs["input_ids"].shape[1]
            text = self.processor.decode(outputs[0][prompt_len:], skip_special_tokens=True)
            logger.info(f"{n}'th answer: {text}")
            preds.append(text)

            if wandb_run is not None:
                wandb_run.log({
                    "step": n,
                    "image": wandb.Image(noisy_image),
                })
                data.append([n, text])
            
            del inputs, outputs
            gc.collect()
            torch.cuda.empty_cache()

        smoothed_pred = self.choose_consistent_prediction(preds, threshold=0.8, sim_model=sim_model)
        logger.info(f"Smoothed prediction: {smoothed_pred}")

        if wandb_run is not None:
            response_table = wandb.Table(columns=columns, data=data)
            wandb.log({"model_responses": response_table})

            final_response_table = wandb.Table(columns=["model_final_response"])
            final_response_table.add_data(smoothed_pred)
            wandb_run.log({"model_final_response": final_response_table})

        return smoothed_pred
