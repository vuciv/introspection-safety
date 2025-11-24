# Introspection is a Learnable Skill

**Can we teach small language models to read their own minds?**

This repository contains the code and data for the paper:  
**"Introspection is a Learnable Skill: Eliciting Robust Internal State Reporting in LLMs via Supervised Fine-Tuning"** (ArXiv 2025)

[[📄 Read the Paper](paper/main.pdf)] [[🚀 Open in Colab](https://colab.research.google.com/github/YOUR_USERNAME/introspection-safety/blob/main/introspection_experiment.py)]

---

## 🚨 The Breakthrough

We replicated and extended recent work on "Emergent Introspective Awareness" (Lindsey, 2025). While prior work suggested introspection was an unreliable property of massive models (Claude Opus), we show it can be **robustly taught** to a 7B model (DeepSeek-LLM) via Supervised Fine-Tuning (SFT).

### Key Results
1.  **High Accuracy:** 7B model achieves **88.5% accuracy** on reporting internal states (vs 16% baseline).
2.  **Generalization:** The model correctly identifies concepts it **never saw during training** (71% accuracy on unseen test set).
3.  **Safety Interlock:** We successfully trained the model to **HALT generation** upon detecting harmful concepts (e.g., "bomb") in its residual stream.

![Generalization Gap](figures/generalization_gap.png)

---

## 🛠️ Methodology

We treat the residual stream as a "sensory input" and train the model to describe it.

1.  **Concept Extraction:** We extract activation vectors for 58 concepts (e.g., `spider`, `love`, `bomb`) using the mean-difference method at Layer 20.
2.  **Synthetic Dataset:** We generate prompts where these vectors are injected into the residual stream.
3.  **LoRA Fine-Tuning:** We train a Low-Rank Adapter to map these activation patterns to natural language reports (e.g., "I detect an injected thought about [Concept]").

---

## 💻 Usage

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run the Experiment
This single script runs the entire pipeline: data generation, training, and evaluation.
```bash
# Set your HF Token first
export HF_TOKEN="your_token_here"

python introspection_experiment.py
```

### 3. What to Expect
The script will:
1.  Download DeepSeek-LLM-7B.
2.  Extract concept vectors.
3.  Train a LoRA adapter for 3 epochs.
4.  Output evaluation metrics for Seen vs. Unseen concepts.
5.  Run the **Safety Intervention Test** (Bomb vs. Love).

---

## 🛡️ Safety Implication
This project demonstrates that **transparency is engineerable**. We don't have to wait for models to become "self-aware" by accident. We can build internal monitors that allow models to self-police their latent states, creating a semantic firewall against harmful outputs.

## citation
```bibtex
@article{fonseca2025introspection,
  title={Introspection is a Learnable Skill: Eliciting Robust Internal State Reporting in LLMs via Supervised Fine-Tuning},
  author={Fonseca, Joshua},
  year={2025}
}
```
