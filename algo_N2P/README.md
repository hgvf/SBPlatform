# Multimodal Stock Forecasting Blueprint

This module proposes a compact-yet-modern multimodal architecture for predicting the next 30 trading-day price trajectory of US equities from (1) the previous 30 trading days of prices and (2) the same-day textual news (headline and article body). The design focuses on:

- Handling *time-series + text* modalities with late-interaction fusion.
- Staying under the 1B parameter budget for initial experimentation.
- Leveraging recent architectural ideas (PatchTST, Q-Formers, diffusion decoders, sparse MoE) while supporting parameter-efficient fine-tuning.
- Operating on the [`oliverwang15/us_stock_news_with_price`](https://huggingface.co/datasets/oliverwang15/us_stock_news_with_price) dataset.

## High-Level Architecture

```
   30-day OHLCV Patch Sequences          Tokenized News (title + body)
                   │                                    │
             PatchTST Encoder                    MiniLM-L6 Encoder
                   │                                    │
         Time-Series Latent Tokens          Textual Latent Tokens (+LoRA)
                   └────────────┬────────────────────────┘
                                │
                        Cross-Modal Q-Former
                                │
                   Diffusion Decoder + Sparse-MoE
                                │
                    30-Day Price Path Samples
```

### Components

1. **PatchTST Time-Series Encoder**  
   - Lightweight patch embedding tailored for financial sequences.  
   - Operates on windowed log-return patches to improve stationarity.  
   - Trains from scratch (no pretraining required).  

2. **MiniLM-L6 Text Encoder with LoRA**  
   - Utilises the `sentence-transformers/all-MiniLM-L6-v2` checkpoint (< 80M parameters).  
   - Applies low-rank adapters on the attention/query/value projections for task-specific conditioning while freezing most weights.  

3. **Cross-Modal Q-Former**  
   - A lightweight query transformer (à la BLIP-2/LLaVA) that learns a fixed set of queries attending jointly to time-series and textual latents.  
   - Produces modality-aware embeddings ready for generative decoding.  

4. **Diffusion Decoder with Sparse-MoE**  
   - Conditional denoising network that predicts future 30-day log-returns.  
   - Composed of stacked Transformer blocks, each featuring a gated sparse Mixture-of-Experts MLP (Top-2 routing).  
   - Supports classifier-free guidance and DDIM sampling to produce price trajectories.  

## Parameter Budget

| Module | Approx. Params | Notes |
| --- | --- | --- |
| PatchTST encoder | ~25M | 8 layers, 128 dims, 8 heads |
| MiniLM-L6 + LoRA | ~80M (frozen) + 2M trainable | LoRA rank=8 |
| Q-Former (8 queries) | ~35M | 12 layers, 256 dims |
| Diffusion decoder + MoE | ~120M | 12 layers, experts with 256→1024→256 |
| **Total** | **≈260M** | comfortably < 1B |

## Training Strategy

1. **Data Preprocessing**  
   - Use dataset splits (`train`/`validation`/`test`) provided by Hugging Face.  
   - For each sample, construct a 30-day rolling window of OHLCV features (close, high, low, open, volume).  
   - Pair with concatenated title+body text from the same reference date.  
   - Forecast target: 30-day future close prices converted to log returns.  

2. **Optimisation**  
   - Freeze MiniLM weights, train LoRA adapters alongside PatchTST, Q-Former, and diffusion decoder.  
   - Optimiser: AdamW (lr=2e-4) with cosine decay; warmup 5%.  
   - Gradient clipping at 1.0, mixed precision (bfloat16) for efficiency.  

3. **Losses & Metrics**  
   - Diffusion objective: mean squared error between predicted and true noise (ε-prediction).  
   - Auxiliary losses: contrastive alignment between modalities (InfoNCE) and price reconstruction (teacher-forced MSE).  
   - Metrics: RMSE/MAE on denormalised prices, directional accuracy, Sharpe ratio of synthetic strategy.  

4. **Fine-Tuning & Extensions**  
   - After baseline convergence, unfreeze MiniLM and continue training with a reduced lr=5e-5.  
   - Optional upgrades: replace MiniLM with FinBERT/FinGPT, swap PatchTST for TimesNet+DLinear hybrid, add CLIP-Adapter or LLaVA-style Q-Former initialisation.  

## Repository Layout

```
multimodal_forecasting/
├── README.md                # Overview & design rationale
├── __init__.py
├── data/
│   └── dataset.py           # Dataset wrappers & preprocessing utilities
├── models/
│   ├── diffusion_decoder.py # Diffusion decoder with sparse-MoE blocks
│   ├── fusion.py            # Cross-modal Q-Former implementation
│   ├── patchtst.py          # Time-series encoder
│   ├── text_encoder.py      # MiniLM + LoRA wrapper
│   └── model.py             # Unified model assembly
└── training/
    ├── config.py            # Configuration dataclasses & defaults
    ├── data.py              # Dataloader helpers for HF datasets
    ├── diffusion.py         # Forward/reverse diffusion utilities
    ├── predict.py           # CLI for inference / sampling
    └── train.py             # CLI for training & validation
```

The accompanying Python modules implement the outlined components and provide a starting point for experimentation and fine-tuning on the chosen dataset.

## Usage

The package exposes command-line entrypoints for training, validation, and inference. Ensure the `src/` directory is on your `PYTHONPATH` (for example, `export PYTHONPATH=src`) and install dependencies as listed in `pyproject.toml`.

### Training

```bash
python -m multimodal_forecasting.training.train \
  --config path/to/experiment.yaml \
  --output-dir runs/exp01 \
  --epochs 5
```

Key flags:

- `--diffusion-steps`: customise the number of diffusion steps (defaults to 1000).
- `--no-eval`: skip validation during training.
- `--validate-only`: run evaluation on a checkpoint without further training.

### Validation Only

```bash
python -m multimodal_forecasting.training.train \
  --checkpoint runs/exp01/checkpoint_best.pt \
  --validate-only
```

### Inference / Sampling

```bash
python -m multimodal_forecasting.training.predict \
  --checkpoint runs/exp01/checkpoint_best.pt \
  --output forecasts.npz \
  --split test \
  --num-samples 8
```

The inference script saves a compressed NumPy archive containing sampled future price trajectories (`predictions`), the ground-truth targets (`targets`), and the associated news text (`texts`).
