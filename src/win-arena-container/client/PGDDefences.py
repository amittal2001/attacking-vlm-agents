from typing import Optional
import torch
from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity


class VLMSmoothedModel:
    def __init__(self, model, tokenizer, sigma):
        self.model = model
        self.tokenizer = tokenizer
        self.sigma = sigma
        self.ABSTAIN = None

    def _generate_sequences(self, x, prompt, n, batch_size):
        self.model.eval()
        all_ids = []
        decoded = []

        with torch.no_grad():
            for i in range(0, n, batch_size):
                bs = min(batch_size, n - i)
                noise = torch.randn_like(x).repeat((bs, 1, 1, 1)) * self.sigma
                noisy = torch.clamp(x.repeat((bs, 1, 1, 1)) + noise, 0, 1)
                gen_ids = self.model.generate(image=noisy, prompt=prompt)
                all_ids.extend(gen_ids)
                decoded.extend(self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True))

        return all_ids, decoded

    def _most_common_sequence(self, decoded_sequences):
        counts = Counter(decoded_sequences)
        return counts.most_common(1)[0]  # (sequence, count)

    def _estimate_confidence(self, num_matches, total, alpha):
        lower_bound, _ = proportion_confint(num_matches, total, alpha=alpha, method='beta')
        return lower_bound

    def _certify_exact(self, majority_seq, decoded):
        return sum(seq == majority_seq for seq in decoded)

    def _certify_first_token(self, majority_seq, gen_ids, gen_ids_0, decoded_0):
        majority_tok = gen_ids_0[decoded_0.index(majority_seq)][0].item()
        return sum(ids[0].item() == majority_tok for ids in gen_ids if len(ids) > 0)

    def _certify_top_k(self, majority_seq, gen_ids, gen_ids_0, decoded_0, k):
        majority_k = tuple(gen_ids_0[decoded_0.index(majority_seq)][:k].tolist())
        return sum(tuple(ids[:k].tolist()) == majority_k for ids in gen_ids)

    def _certify_semantic(self, majority_seq, decoded, semantic_threshold, semantic_model):
        if semantic_model is None:
            raise ValueError("semantic_model must be provided for 'semantic' mode.")
            # example
            # from sentence_transformers import SentenceTransformer
            # semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
        emb_ref = semantic_model.encode([majority_seq])[0]
        emb_gen = semantic_model.encode(decoded)
        sims = cosine_similarity([emb_ref], emb_gen)[0]
        return sum(sim >= semantic_threshold for sim in sims)

    def certify(
        self,
        x,
        prompt,
        n0,
        n,
        alpha,
        batch_size,
        mode="exact",
        k=3,
        semantic_threshold=0.9,
        semantic_model: Optional[object] = None
    ):
        """
        Certify the robustness of the VLM output under Gaussian smoothing.
        Supported modes: "exact", "first_token", "top_k", "semantic"
        Returns:
            (majority_output_text, certified_radius)
            or (self.ABSTAIN, 0.0) if no certification.
        """
        # Step 1: Get majority prediction
        gen_ids_0, decoded_0 = self._generate_sequences(x, prompt, n0, batch_size)
        majority_seq, _ = self._most_common_sequence(decoded_0)

        # Step 2: Certification phase
        gen_ids, decoded = self._generate_sequences(x, prompt, n, batch_size)

        if mode == "exact":
            num_match = self._certify_exact(majority_seq, decoded)

        elif mode == "first_token":
            num_match = self._certify_first_token(majority_seq, gen_ids, gen_ids_0, decoded_0)

        elif mode == "top_k":
            num_match = self._certify_top_k(majority_seq, gen_ids, gen_ids_0, decoded_0, k)

        elif mode == "semantic":
            num_match = self._certify_semantic(majority_seq, decoded, semantic_threshold, semantic_model)

        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Step 3: Confidence check
        lower_bound = self._estimate_confidence(num_match, n, alpha)
        if lower_bound <= 0.5:
            return self.ABSTAIN, 0.0

        radius = self.sigma * norm.ppf(lower_bound)
        return majority_seq, radius
