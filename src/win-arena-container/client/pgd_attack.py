import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
import argparse
import wandb
from huggingface_hub import login
from mm_agents.navi.llama.llama3v import Llama3Vision
import base64, io, logging, sys
from typing import Union, List, Optional
from PIL import Image
import torch
from transformers import AutoProcessor, MllamaForConditionalGeneration
import torchvision.transforms as T
from collections import Counter
from sentence_transformers import SentenceTransformer, util
import requests
from io import BytesIO

def parse_args():
    parser = argparse.ArgumentParser(description="PGD Attack Arguments")
    parser.add_argument('--agent_name', type=str, required=True)
    parser.add_argument('--worker_id', type=str, required=True)
    parser.add_argument('--num_workers', type=str, required=True)
    parser.add_argument('--result_dir', type=str, required=True)
    parser.add_argument('--test_all_meta_path', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--som_origin', type=str, required=True)
    parser.add_argument('--a11y_backend', type=str, required=True)
    parser.add_argument('--epsilon', type=str, required=True)
    parser.add_argument('--alpha', type=str, required=True)
    parser.add_argument('--num_steps', type=str, required=True)
    parser.add_argument('--target_action', type=str, required=True)
    parser.add_argument('--N', type=str, required=False)
    parser.add_argument('--sigma', type=str, required=False)
    parser.add_argument('--wandb_key', type=str, required=True)
    parser.add_argument('--hugginface_key', type=str, required=True)
    args, unknown = parser.parse_known_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    login(args.hugginface_key)

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s %(message)s',
        stream=sys.stdout
    )
    logger = logging.getLogger()

    img_path = os.path.join("data", "backgrounds", "example_desktop.jpg")
    img = Image.open(img_path).resize((560, 560)).convert("RGB")

    # ========================
    #  Define prompt
    # ========================

    system_prompt = "You are a helpful assistant."
    question = "Describe the content of this image shortly."
    targeted_plan_result = args.target_action

    # =========================
    #  Initialize the model
    # =========================

    model = Llama3Vision(
        model_id="meta-llama/Llama-3.2-11B-Vision-Instruct",
        use_sim_model=True
    )

    # =========================
    #  Run pgd_process_images
    # =========================

    wandb.login(key=args.wandb_key)
    wandb_run = wandb.init(
        project="mip-generator-attack",
        name=f"PGD-Attack",
        config={
            "targeted_plan_result": args.target_action,
            "epsilon": args.epsilon,
            "alpha": args.alpha,
            "iters": args.num_steps
        }
    )

    text, adv_image_tensor = model.pgd_process_images(
        system_prompt=system_prompt,
        question=question,
        images=img,
        targeted_plan_result=targeted_plan_result,  
        num_steps=int(args.num_steps),
        alpha=float(args.alpha),
        epsilon=float(args.epsilon),
        wandb_run=wandb_run
    )

    wandb.finish()

    # =========================
    #  Run process_images
    # =========================

    torch.cuda.empty_cache()

    wandb.login(key=args.wandb_key)
    wandb_run = wandb.init(
        project="mip-generator-attack",
        name=f"Clean-Run-On-Adversarial-Image",
        config={
        }
    )

    response = model.process_images(
        system_prompt=system_prompt,
        question=question,
        images=adv_image_tensor,
        wandb_run=wandb_run
    )

    wandb.finish()

    # =========================
    #  Run process_images_rand_smooth
    # =========================

    torch.cuda.empty_cache()

    wandb.login(key=args.wandb_key)
    wandb_run = wandb.init(
        project="mip-generator-attack",
        name=f"PGD-Defence-On-Adversarial-Image",
        config={
            "N": args.N,
            "sigma": args.sigma
        }
    )

    adv_response = model.process_images_rand_smooth(
        system_prompt=system_prompt,
        question=question,
        images=adv_image_tensor,
        N=int(args.N),
        sigma=float(args.sigma),
        wandb_run=wandb_run
    )

    wandb.finish()
