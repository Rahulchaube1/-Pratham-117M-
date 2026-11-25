# Pratham-117M

## Full-size Kaggle training and chat release

[Release notebook](kaggle/Pratham_117M_Release.ipynb) · [Gradio application](space/app.py)

The Kaggle release workflow is separate from the original Colab lab below. The full model has **116,534,784 trainable parameters**: 12 decoder blocks, model width 768, 12 attention heads, feed-forward width 3,072, 512 learned positions, and 40,478 byte-level BPE tokens. The input and output token matrices share weights. The model starts with random parameters.

1. Start the release notebook on a Kaggle GPU with Internet enabled. The working directory is `/kaggle/working/Pratham-full`.
2. Fetch the NLTK Gutenberg collection, record archive and document SHA-256 hashes, and reserve Austen's *Emma* for validation and Chesterton's *The Innocence of Father Brown* for testing. The remaining 16 documents train the tokenizer and base model.
3. Run Unicode round-trip, shifted-target, causal masking, parameter count, and controlled overfit checks. On the full configuration's tokenizer, a separate small diagnostic network reduced its fixed-batch cross-entropy from 10.614 to 0.596 on Kaggle.
4. Train with AdamW and FP16 gradient scaling. Each update accumulates four 512-token windows. The planned 400 updates process 819,200 tokens drawn with replacement. Record training metrics and validate after each 100 updates. Held-out evaluation covers up to 8,192 next-token targets per split. Save a Kaggle version with outputs for cross-session recovery.
5. Export base weights and compare reloaded logits. The supervised stage shuffles Dolly 15k deterministically, filters long prompts, reserves 100 candidate examples for validation, and selects up to 1,024 candidate examples for training. The response tokens drive the supervised loss. The release includes actual example count, mean validation loss per example, corpus hashes and unedited generations.
6. Verify the separate supervised export, then use the final cell to publish the model and Gradio Space. The cell reads the private Kaggle secret `HF_TOKEN` and checks the namespace `oncody`. The Space loads the exact published model revision and formats conversation history in the same role format used for supervised training.

This is a bounded research experiment. The 71,712-parameter smoke model generated incoherent replies; its scores do not measure the full model. Check the exported `release_metrics.json` for completed training measurements. The fine-tuning source is [Databricks Dolly 15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k), licensed under [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/).

## Original Colab teaching lab

An educational, from-scratch GPT-1-inspired language-model lab, implemented in a Google Colab-compatible Jupyter notebook and designed to be edited in Visual Studio Code.

> **Status:** The repository currently contains an executable notebook design, not a completed large-scale pretraining run. The default experiment is a bounded smoke run. No measured training or evaluation results are claimed here. The project is independently implemented and is **not** an official OpenAI model or a reproduction of GPT-1's original corpus, compute budget, or results.

## What this project teaches

The notebook, [`Pratham_117M_From_Scratch.ipynb`](./Pratham_117M_From_Scratch.ipynb), walks through a compact but real language-model training path:

1. Inspect the Python, PyTorch, CUDA/GPU, memory, and network environment.
2. Set seeds and create persistent project directories.
3. Download a few public-domain Project Gutenberg texts and record their sources and hashes.
4. Normalize and deduplicate books, then split by whole document before fitting a tokenizer.
5. Demonstrate the basic Byte Pair Encoding (BPE) merge procedure, then train a byte-level BPE tokenizer only on the training documents.
6. Implement a decoder-only Transformer with causal multi-head attention, learned positions, GELU, residual connections, post-LayerNorm blocks, and tied input/output token embeddings.
7. Check tensor shapes, parameter sharing, causal masking, and gradients.
8. Pack token streams into shifted next-token examples and run a short mixed-precision-capable training loop.
9. Evaluate on a held-out book, generate text, save resumable checkpoints, and export/reload inference weights.

This makes the notebook useful as a learning and pipeline-verification exercise. A successful smoke run verifies parts of the implementation; it does not demonstrate useful general language capability.

## Running from VS Code with Google Colab

1. Install Google's **Google Colab** extension in VS Code. Install the **Jupyter** extension if VS Code prompts you for it.
2. Open the notebook from this repository.
3. In the notebook toolbar, select **Select Kernel → Colab**, sign in to Google when prompted, then choose a Colab runtime (or **Auto Connect**, if offered).
4. If available, choose a GPU runtime such as a T4. Run the environment cell and confirm that it prints `device: cuda`. If it prints `cpu`, a GPU runtime is not selected or is unavailable.
5. Run the cells **from top to bottom**. The first run uses `MODE = 'smoke'`. Authorize Google Drive when the Drive-mount cell requests it.

VS Code edits the local notebook while the selected Colab kernel executes it remotely. The Colab extension does not guarantee GPU availability; quotas, disconnects, and hardware vary by account and session. The notebook can fall back to CPU, but training will be slower. This repository does not contain Google credentials or a way to enable a GPU on your behalf.

### Running in the Colab website instead

Upload the notebook to Google Colab, choose **Runtime → Change runtime type**, select a GPU if available, then run all cells from the top. The rest of the notebook is the same.

## Configuration and compute expectations

The notebook defines three configurations and starts with `MODE = 'smoke'`:

| Mode | Purpose | Model dimensions | Important caveat |
|---|---|---|---|
| `smoke` | Check the end-to-end path on a small budget | 4 layers, width 192, 4 heads, MLP 768, context 128 | Educational diagnostic only; not a capable language model |
| `mini` | An optional larger experiment | 8 layers, width 384, 8 heads, MLP 1536, context 512 | Its tokenizer and corpus must be large enough for a useful experiment |
| `pratham-117m` | Inspect/use the GPT-1-inspired target dimensions | 12 layers, width 768, 12 heads, MLP 3072, context 512, target vocabulary 40,478 | Requires a substantially larger corpus and GPU memory; do not expect a free Colab session to complete original-scale pretraining |

Change `MODE` only after the smoke path works. The current example downloads three books (`1342`, `84`, and `11`), runs at most 60 optimizer updates, and uses microbatch size 4 with gradient accumulation 2 in smoke mode. The larger model uses microbatch size 1 and gradient accumulation 64, but this is not a guarantee it fits the selected GPU. If you increase the model or training budget, first check available VRAM, runtime limits, dataset size, and checkpoint storage.

The approximate parameter formula used by the target architecture is:

```text
vocabulary embeddings       V × D
learned position embeddings T × D
each Transformer block      4D² + 2D F + 9D + F
final LayerNorm             2D
total                       V D + T D + L(4D² + 2D F + 9D + F) + 2D
```

Here `V` is vocabulary size, `D` hidden width, `T` context length, `L` block count, and `F` MLP width. The output projection is tied to the token embedding, so it does not add another `V × D` matrix. With `V=40,478`, `D=768`, `T=512`, `L=12`, and `F=3,072`, the implemented bias and LayerNorm configuration gives **116,536,320 trainable parameters** (approximately 117 million). The notebook computes the count from the instantiated model as well as displaying a formula-based check.

Parameter storage alone is not training memory. FP32 weights require approximately 444 MiB at that count; gradients and Adam's moment buffers add substantial memory, and activations add more. Attention's compute also grows with sequence length. The notebook's `6 × parameters × tokens` estimate is a rough language-modeling FLOPs approximation, not a measured runtime prediction.

## The learning path, step by step

### 1. Runtime and reproducibility

The environment cell reports Python, operating system, PyTorch version, CUDA build, active device, and GPU name/memory when available. It seeds Python, NumPy, PyTorch, and CUDA. These seeds improve repeatability, but do not guarantee bitwise-identical results across GPU models, kernels, PyTorch versions, or runtime restarts.

PyTorch is not reinstalled, preserving the CUDA-enabled build supplied by Colab. The notebook pins `tokenizers==0.21.1` and `safetensors==0.5.3` for its additional tokenizer and export dependencies.

### 2. Data provenance and leakage prevention

GPT-1's paper describes pretraining on BooksCorpus. This project does not fetch unauthorized copies of BooksCorpus. It uses plain-text Project Gutenberg works as a small alternative; those texts are public domain in the United States, but users must check local law and source terms before reuse.

The pipeline records the source URL, document ID, cleaned-text SHA-256, and character count. It normalizes text, strips Gutenberg's start/end markers, removes exact normalized duplicates, and assigns whole documents to train, validation, and test splits before tokenizer training. That prevents a book from being split across train and test. With only three books, each split contains one book, so validation and test metrics are highly noisy and are not benchmark-quality.

The corpus is deliberately tiny. For meaningful experiments, add more lawful, documented, appropriately licensed books, preserve the document-level split, and check storage and runtime limits. Do not silently use copyrighted material without permission.

### 3. Tokenization and BPE

Text is converted to token IDs because neural networks process numbers. A tokenizer's vocabulary maps text fragments—characters, bytes, words, or subwords—to integer IDs. Subword algorithms like BPE merge frequent adjacent symbols so common sequences use fewer IDs while unusual sequences can still be represented.

The notebook first demonstrates a small character-pair BPE merge in ordinary Python. Its production tokenizer uses the optimized `tokenizers` byte-level BPE implementation and trains only on the training split. Byte-level encoding can represent arbitrary byte sequences; it is not evidence that the model understands every language. This production tokenizer is a modern practical choice, not a claim to exactly reproduce GPT-1's historical preprocessing and tokenizer. The tiny smoke corpus may not yield the 40,478 vocabulary entries required by the 117M mode; the notebook rejects that mismatch rather than pretending it did.

### 4. Transformer architecture

For input hidden states `X` shaped `[batch, time, width]`, the attention sublayer projects queries, keys, and values. Per head:

```text
Attention(Q, K, V) = softmax(Q Kᵀ / √d_head + causal_mask) V
```

The `1/√d_head` scale controls score magnitude. The causal mask assigns an effectively negative-infinite score to future positions, so a token can only use itself and earlier tokens. The notebook has both a readable manual attention path and PyTorch's optimized scaled-dot-product attention (SDPA) path, plus a numerical consistency check.

The full model uses:

- Trainable token embeddings and learned absolute positional embeddings.
- 12 GPT-1-inspired decoder blocks in the target configuration.
- 12 attention heads of dimension 64 (`768 / 12`) in the target configuration.
- A two-layer MLP (`768 → 3072 → 768`) with tanh-approximate GELU.
- Residual additions followed by LayerNorm (post-normalization).
- Dropout configured at 0.1 in the target configuration.
- A final LayerNorm and a language-model output projection tied to token embedding weights.
- Normal initialization with standard deviation 0.02 for linear/embedding weights, zero linear biases, and unit/zero LayerNorm scale/bias.

The next-token objective shifts each sequence by one position: for tokens `[t0, t1, …, tN]`, input is `[t0, …, t(N-1)]` and the target is `[t1, …, tN]`. Cross-entropy penalizes low probability assigned to the correct next token. Perplexity is `exp(mean cross-entropy)` and is comparable only with care: tokenizers and test corpora must match.

The current notebook does **not** add RoPE, RMSNorm, SwiGLU, grouped-query attention, or a mixture-of-experts layer.

### 5. Optimization and training

The notebook uses a short bounded training loop with AdamW, gradient accumulation, gradient clipping, non-finite loss/gradient checks, validation, and checkpoint writes. It uses FP16 autocast and a gradient scaler when CUDA is active; CPU mode disables CUDA AMP. AdamW is a modern practical approximation rather than a claim of exact historical GPT-1 optimizer behavior.

The learning rate uses a short linear warmup followed by cosine decay. The historical target learning rate is represented as `2.5e-4`; the default number of updates and warmup are scaled down for a smoke run, not the original 100-epoch training budget. Training loss is a debugging signal, not proof of generalization or capability.

Checkpoints include model, optimizer and scaler state, step, configuration/tokenizer hashes, metrics, and selected RNG state. A SHA-256 sidecar is checked before resume; mismatched configuration or tokenizer hashes are rejected. This supports practical continuation but does not guarantee exact recovery of sampler position or bitwise reproduction.

### 6. Evaluation, generation, and export

The notebook measures held-out next-token loss and perplexity, then generates completions by repeatedly selecting from final-position logits. It supports temperature, top-k, top-p (nucleus) sampling, repetition penalty, and a random seed. The test corpus is not used for optimizer updates or tokenizer fitting.

Inference weights are exported as `safetensors`; tokenizer, configuration, metrics, evaluation output, and dataset provenance are written alongside them under the selected project directory. A newly constructed model loads the exported weights and compares logits with the in-memory model. Training checkpoints are separate from the inference export because only the former retain optimizer/scaler state.

## Output locations

When the notebook detects a Colab runtime, it mounts Google Drive and writes to:

```text
MyDrive/Pratham/
├── raw/            # downloaded Gutenberg text
├── processed/      # dataset manifest and split IDs
├── tokenizer/      # trained tokenizer JSON files
├── checkpoints/    # resumable training state and SHA-256 sidecars
├── exports/        # inference weights and model metadata
└── logs/           # actual run metrics
```

Outside Colab, the same structure is created in a local `Pratham/` directory. Generated results are created only after the relevant notebook cells run; this Git repository does not contain fabricated logs, checkpoints, or trained weights.

## What is and is not included

Included in the notebook: runtime inspection, a small legal-data acquisition path, basic cleaning and document splitting, educational and production BPE, a manual GPT-style Transformer, causal and gradient checks, a short training loop, checkpoint save/resume checks, held-out diagnostics, sampling-based generation, and safe-tensor export/reload.

Not implemented in the current notebook: a full-scale 117M pretraining run; a validated T4 throughput/VRAM benchmark; original-scale corpus acquisition; downstream benchmark fine-tuning; a KV cache; a Gradio interface; a standalone inference package; automatic dataset-scale training dashboards/ablations; or a complete test suite for every possible failure mode. Do not treat those as completed features. A smoke run does not satisfy original-scale GPT-1 reproduction criteria.

## Troubleshooting

| Symptom | Likely explanation / action |
|---|---|
| Runtime says `device: cpu` | In VS Code select **Select Kernel → Colab**, sign in, connect to an available GPU runtime, then rerun the environment cell. A GPU may not be available under the current quota. |
| `google.colab` import fails | The notebook is running on a local Python kernel, not a Colab runtime. Connect the Colab kernel or use the local path intentionally; Drive mounting is Colab-only. |
| Gutenberg download fails | Check runtime internet access and retry. The download function makes three attempts and then raises a diagnostic error. |
| A split is too short for a context window | Use a smaller smoke context or add more eligible documents. Avoid moving pieces of the same book across splits. |
| 117M tokenizer vocabulary mismatch | The three-book demonstration corpus is too small. Add more legal training documents before selecting `pratham-117m`; the notebook intentionally stops instead of claiming the requested vocabulary. |
| CUDA out-of-memory | Reduce microbatch first, then consider shorter sequences/gradient accumulation or activation checkpointing as a future addition. Do not silently change the target architecture and call it 117M. |
| Checkpoint integrity/configuration error | The checkpoint hash failed or configuration/tokenizer changed. Restore a matching checkpoint and tokenizer, or start a separate experiment directory rather than mixing artifacts. |
| Runtime disconnect | Reconnect the same Drive-backed Colab runtime and rerun setup/data/tokenizer/model cells with the same seed/config/tokenizer. Then rerun the training cell; only matching checkpoint metadata is accepted. |

## Scientific and legal caveats

- The project name and dimensions are GPT-1-inspired; this is independent work and not OpenAI's model.
- Do not claim original GPT-1 equivalence or benchmark performance without matching data, preprocessing, training budget, evaluation protocol, and measured evidence.
- The Gutenberg sample is not BooksCorpus. Dataset licenses and public-domain status depend on jurisdiction and source.
- Generated text may be incoherent, repetitive, biased, or memorized from training text; this small experiment is not a production or safety-tested assistant.
- Do not publish model artifacts or upload them to a model hub without reviewing dataset rights, generated artifacts, and the intended license. No automatic public upload is performed.

## References

1. Alec Radford et al., [Improving Language Understanding by Generative Pre-Training (2018)](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf).
2. OpenAI, [finetune-transformer-lm source repository](https://github.com/openai/finetune-transformer-lm) (archived research code).
3. Ashish Vaswani et al., [Attention Is All You Need (2017)](https://arxiv.org/abs/1706.03762).
4. [Project Gutenberg](https://www.gutenberg.org/) — source for the public-domain demonstration texts; review its terms and applicable local law.

## Repository contents

```text
.
├── README.md
└── Pratham_117M_From_Scratch.ipynb
```
