# OpenGVL Integration Guide for RFM Models

This guide explains how to integrate your trained RFM (Reward Foundation Model) models with OpenGVL for evaluation.

## Overview

OpenGVL is a benchmarking framework for visual temporal progress prediction. This integration allows you to:
1. Use your trained RFM models within the OpenGVL evaluation framework
2. Evaluate RFM models on standard datasets using OpenGVL's metrics (e.g., VOC score)
3. Compare RFM performance with other vision-language models

## Prerequisites

1. **OpenGVL submodule**: The `opengvl` directory should be present as a git submodule in your `reward_fm` repository.
2. **RFM model checkpoint**: A trained RFM model checkpoint (HuggingFace repo ID or local path).
3. **Python environment**: Ensure both `reward_fm` and `opengvl` dependencies are installed.

### Installing Dependencies

The RFM package is included as a dependency in OpenGVL's `pyproject.toml`. To install:

```bash
cd /scr/aliang80/reward_fm/opengvl
uv sync
```

This will install OpenGVL and its dependencies, including the RFM package (as a local editable dependency).

## Step 1: Verify Submodule Setup

First, ensure the OpenGVL submodule is properly initialized:

```bash
cd /scr/aliang80/reward_fm
git submodule update --init --recursive
```

## Step 2: Configure Your RFM Model

Edit the RFM model configuration file:

```bash
# Edit the model config
vim opengvl/configs/model/rfm.yaml
```

Update the `model_path` field with your RFM checkpoint path:

```yaml
_target_: opengvl.clients.rfm.RFMClient
model_path: "your-username/your-rfm-model"  # HuggingFace repo ID
# OR
model_path: "/path/to/local/checkpoint"  # Local path
rpm: 0.0  # No rate limit for local models
```

## Step 3: Add Your Evaluation Datasets

OpenGVL uses dataset configurations to specify which datasets to evaluate on. We've created example configs for some of your eval datasets.

### Existing Dataset Configs

The following dataset configs are already created:
- `opengvl/configs/dataset/metaworld_eval.yaml`
- `opengvl/configs/dataset/libero_rfm.yaml`
- `opengvl/configs/dataset/oxe_rfm_eval.yaml`

### Creating New Dataset Configs

To add a new dataset, create a YAML file in `opengvl/configs/dataset/`:

```yaml
# opengvl/configs/dataset/your_dataset.yaml
name: your_dataset_name
dataset_name: "huggingface-org/dataset-name"  # HuggingFace dataset identifier
camera_index: 0  # Which camera view to use (usually 0)
max_episodes: 50  # Maximum number of episodes to evaluate
num_frames: 15  # Number of frames per episode
num_context_episodes: 2  # Number of few-shot context episodes
```

**Important**: The dataset must be a LeRobot-compatible dataset hosted on HuggingFace. 

**Note**: Your RFM datasets may need to be converted to LeRobot format. See `DATASET_CONVERSION_GUIDE.md` for detailed instructions on converting RFM datasets to LeRobot format.

### Dataset Requirements

For OpenGVL to work with your datasets, they need to:
1. Be hosted on HuggingFace Hub
2. Follow LeRobot dataset format (with `episodes` and camera keys)
3. Include task instructions in the `task` field
4. Have frames accessible via camera keys

## Step 4: Run Evaluation

### Basic Evaluation

Run evaluation with your RFM model:

```bash
cd /scr/aliang80/reward_fm/opengvl
uv run python -m opengvl.scripts.predict \
  --config-dir configs/experiments \
  --config-name predict \
  model=rfm \
  dataset=metaworld_eval \
  prediction.num_examples=50
```

### Customizing Evaluation

You can override any configuration parameter from the command line:

```bash
cd /scr/aliang80/reward_fm/opengvl
uv run python -m opengvl.scripts.predict \
  --config-dir configs/experiments \
  --config-name predict \
  model=rfm \
  model.model_path="your-model-path" \
  dataset=metaworld_eval \
  dataset.max_episodes=100 \
  prediction.num_examples=50 \
  prediction.temperature=1.0
```

### Key Configuration Parameters

- `model.model_path`: Path to your RFM checkpoint
- `dataset.name`: Dataset identifier (must match a config in `configs/dataset/`)
- `prediction.num_examples`: Number of episodes to evaluate
- `dataset.num_frames`: Number of frames per episode
- `dataset.num_context_episodes`: Number of few-shot examples (RFM doesn't use these, but they're included for compatibility)

## Step 5: Understanding Outputs

OpenGVL generates two main output files:

1. **Predictions JSONL** (`{model_name}_{timestamp}_predictions.jsonl`):
   - Contains per-episode predictions and metrics
   - Includes raw model outputs (if `prediction.save_raw=true`)

2. **Summary JSON** (`{model_name}_{timestamp}_summary.json`):
   - Aggregated metrics across all episodes
   - Includes VOC (Visual Ordering Consistency) score
   - Dataset and model metadata

### Output Location

By default, outputs are saved to:
- `opengvl/results/` (single run)
- `opengvl/multirun/` (Hydra multirun/sweep)

You can customize this:
```bash
prediction.output_dir="/path/to/output"
```

## Step 6: Adding More Datasets

To add additional datasets from your eval list:

1. **Identify the HuggingFace dataset name**: Check your `rfm/configs/preprocess.yaml` for dataset names.

2. **Create a config file**:
   ```bash
   # Example: opengvl/configs/dataset/libero_failure.yaml
   name: libero_failure
   dataset_name: "ykorkmaz/libero_failure_rfm"
   camera_index: 0
   max_episodes: 50
   num_frames: 15
   num_context_episodes: 2
   ```

3. **Run evaluation**:
   ```bash
   cd /scr/aliang80/reward_fm/opengvl
   uv run python -m opengvl.scripts.predict \
     --config-dir configs/experiments \
     --config-name predict \
     model=rfm \
     dataset=libero_failure
   ```

## Troubleshooting

### Import Errors

If you see import errors for RFM modules:
- Ensure you're using `uv run` from the `opengvl` directory (this sets up the correct environment)
- Check that the `rfm` package is installed/accessible via the dependency in `opengvl/pyproject.toml`

### Model Loading Issues

- Verify the `model_path` points to a valid checkpoint
- For HuggingFace repos, ensure you have access (set `HUGGING_FACE_HUB_TOKEN` if needed)
- Check that the model checkpoint includes `config.yaml`

### Dataset Loading Issues

- Verify the dataset exists on HuggingFace Hub
- Check that the dataset follows LeRobot format
- Ensure `HUGGING_FACE_HUB_TOKEN` is set if the dataset is private

### Frame Format Issues

- RFM expects frames as numpy arrays (H, W, C) with uint8 dtype
- OpenGVL handles conversion automatically, but if you see errors, check frame shapes

## Advanced Usage

### Batch Evaluation Across Multiple Datasets

Create a sweep configuration or use Hydra's multirun:

```bash
cd /scr/aliang80/reward_fm/opengvl
uv run python -m opengvl.scripts.predict \
  --config-dir configs/experiments \
  --config-name predict \
  model=rfm \
  dataset=metaworld_eval,libero_rfm,oxe_rfm_eval \
  -m
```

### Custom Prompt Templates

You can customize prompts by modifying `configs/prompts/` and `configs/prompt_phrases/`. However, RFM doesn't use text prompts, so these won't affect RFM predictions.

### Comparing with Other Models

To compare RFM with other models (e.g., Gemini, Qwen):

```bash
cd /scr/aliang80/reward_fm/opengvl

# Run with RFM
uv run python -m opengvl.scripts.predict \
  --config-dir configs/experiments \
  --config-name predict \
  model=rfm \
  dataset=metaworld_eval

# Run with Gemini (requires API key)
uv run python -m opengvl.scripts.predict \
  --config-dir configs/experiments \
  --config-name predict \
  model=gemini \
  dataset=metaworld_eval
```

Then compare the summary JSON files.

## Architecture Notes

### How RFM Integration Works

1. **RFMClient** (`opengvl/clients/rfm.py`):
   - Inherits from `BaseModelClient`
   - Loads your RFM model checkpoint on initialization
   - Overrides `_generate_response_impl` to use RFM's `compute_progress` method
   - Converts RFM's 0-1 progress values to 0-100 percentages for OpenGVL

2. **Data Flow**:
   - OpenGVL loads episodes from HuggingFace datasets
   - Frames are extracted and shuffled (as per OpenGVL's evaluation protocol)
   - RFMClient receives the episode with frames and task instruction
   - RFM model predicts progress for each frame
   - Predictions are formatted as text (for compatibility with OpenGVL's mapper)
   - OpenGVL's mapper extracts percentages and computes metrics

3. **Key Differences from VLM Models**:
   - RFM doesn't use few-shot context (context episodes are ignored)
   - RFM doesn't use text prompts (only task instruction)
   - RFM returns deterministic predictions (no temperature sampling)

## Next Steps

1. **Evaluate on your datasets**: Add configs for all your eval datasets and run evaluations
2. **Compare metrics**: Compare VOC scores and other metrics across different models
3. **Analyze results**: Use the JSONL outputs for detailed per-episode analysis
4. **Contribute back**: Consider contributing your dataset configs or improvements to the RFM client

## References

- OpenGVL repository: https://github.com/budzianowski/opengvl
- OpenGVL paper: Check the repository for citation details
- RFM documentation: See `rfm/` directory in your reward_fm repository

