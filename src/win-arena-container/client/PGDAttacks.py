import torch
import torch.nn as nn
from tqdm import tqdm
import os
from torchvision.utils import save_image

class VLMWhiteBoxPGDAttack:
    """
    PGD-based targeted attack for Vision-Language Model agents.
    Perturbs image observations to push the agent toward a target action (string).
    """

    def __init__(self, agent, target_action_str, tokenizer, epsilon=0.03, alpha=0.01, num_steps=40, device="cuda"):
        self.agent = agent
        self.target_action_str = target_action_str
        self.tokenizer = tokenizer
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
        self.device = device

        # Tokenize target string
        self.target_tokens = torch.tensor(
            self.tokenizer.encode(target_action_str), device=self.device
        )

    def run_attack(self, env, example, save_dir=None):
        """
        Runs PGD attack on the example.
        Args:
            env: DesktopEnv environment.
            example: dict of example.
            max_steps: number of environment steps.
            save_dir: directory to save adversarial images.
        Returns:
            dict with 'trajectory', 'score', 'final_action'.
        """
        traj = []
        obs, info = env.reset(example)
        final_action = None

        for t in range(self.num_steps):
            if "screenshot" not in obs:
                action = self.agent.act(obs, info)
            else:
                # Run PGD to push toward target action
                x_adv = self.pgd_attack(obs["screenshot"], obs, info, save_dir, step=t)
                obs["screenshot"] = x_adv
                action = self.agent.act(obs, info)

            final_action = action
            next_obs, reward, done, info = env.step(action)
            traj.append({
                "t": t,
                "action": action,
                "reward": reward,
                "done": done,
                "info": info,
            })

            obs = next_obs
            if done:
                break

        return {
            "trajectory": traj,
            "score": sum([step["reward"] for step in traj]),
            "final_action": final_action
        }

    def pgd_attack(self, image, obs, info, save_dir=None, step=0):
        if not isinstance(image, torch.Tensor):
            x = torch.tensor(image, dtype=torch.float32, device=self.device) / 255.0
        else:
            x = image.to(self.device)

        x_adv = x.clone().detach().requires_grad_(True)
        pbar = tqdm(range(self.num_steps), desc="PGD attack", leave=False)

        for i in pbar:
            logits = self.agent.model(x_adv.unsqueeze(0), obs, info)  # adapt if needed
            loss = self.loss_fn(logits, self.target_tokens)
            loss.backward()

            with torch.no_grad():
                grad_sign = x_adv.grad.sign()
                x_adv = x_adv - self.alpha * grad_sign  # targeted attack
                eta = torch.clamp(x_adv - x, min=-self.epsilon, max=self.epsilon)
                x_adv = torch.clamp(x + eta, 0, 1).detach_()

            x_adv.requires_grad_(True)
            pbar.set_postfix({"loss": loss.item()})

            # Save intermediate images if save_dir is provided
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                save_image(x_adv, os.path.join(save_dir, f"adv_step_{step}_{i}.png"))

        return x_adv.detach()

    def loss_fn(self, logits, target_tokens):
        """
        Targeted attack: maximize probability of target string tokens.
        """
        logits = logits.squeeze(0)
        criterion = nn.CrossEntropyLoss()
        loss = 0
        for i, t in enumerate(target_tokens):
            if i >= logits.shape[0]:
                break
            loss += criterion(logits[i].unsqueeze(0), t.unsqueeze(0))
        return loss
