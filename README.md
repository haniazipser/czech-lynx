# czech-lynx

## Project Structure
```
.
├── config            # Config files w presets per model
├── data              # Datasets
├── evaluation        # Metrics & evaluation
├── models            # Model definitions
├── run               # Per run visualization and model checkpoints (not tracked)
├── scripts           # Data exploration scripts
├── tests             # Scripts for inference on test dataset
├── training          # Training pipeline
├── utils             # Shared utilities
├── visualisations    # General Graphs about data 
├── wildfusion        # Fusion pipeline
```
## Correct Kaggle Dataset Download Flow
Step 1: Install Kaggle API
``` bash
pip install kaggle
```
---
Step 2: Authenticate 

- Go to Kaggle → Account → API

- Generate legacy API Token

- You receive `kaggle.json` 

- Place token in:

    ```C:\Users\<USER>\.kaggle\kaggle.json```
    
(create .kaggle if it does not exist)

---
Step 3: Download dataset
``` bash
kaggle datasets download -d picekl/czechlynx -p data/kaggle-data --unzip
```
---
Step 4. Correct Data Structure
After extraction:

    data/kaggle-data
     ├── CzechLynx/
     ├── CzechLynx_Sythetic/
     ├── CzechLynxDataset-Metadata-Real.csv
     ├── CzechLynxDataset-Metadata-Synthetic.csv
---


## W&B login

If you want experiment tracking with Weights & Biases, login first:

```bash
wandb login
```

Then follow the URL and paste your API key when prompted.

If you prefer environment variables, set:

```bash
export WANDB_API_KEY=your_api_key_here
```

## Visualizing dataset

```bash
python explore_data.py 
```

# Training Guide

## Baseline Training

To start a new baseline training run:

```bash
python run_baseline.py
```

To resume from an existing run and generate visualizations from a saved checkpoint:

```bash
python run_baseline.py --run-id <RUN_ID>
```

The script will automatically load the corresponding checkpoint and create the visualizations.

---

## MegaDescriptor Training

To train the MegaDescriptor model:

```bash
python run_reid.py
```

The script supports the same `--run-id` argument for resuming training or loading an existing checkpoint:

```bash
python run_reid.py --run-id <RUN_ID>
```

---

## Full WildFusion Training

Training the complete WildFusion pipeline requires a pre-trained MegaDescriptor model.

### Step 1: Generate LightGlue Training Data

Run:

```bash
python get_data_lightglue.py
```

This generates the initial training data required for the next stage.

### Step 2: Generate MegaDescriptor Features

Using the run ID of a trained MegaDescriptor model, execute:

```bash
python get_data_megadescriptor.py --run-id <RUN_ID>
```

This process creates a CSV file containing the features and labels required for calibrator training.

### Step 3: Train the Calibrator

Once the CSV dataset has been generated, start calibrator training:

```bash
python run_calibrator.py
```

After completion, the calibrated model can be used as part of the full WildFusion pipeline.

---

## Training Pipeline Summary

1. Train a MegaDescriptor model (`run_reid.py`).
2. Generate LightGlue training data (`get_data_lightglue.py`).
3. Generate MegaDescriptor features using the trained MegaDescriptor run ID (`get_data_megadescriptor.py`).
4. Train the calibrator (`run_calibrator.py`).
5. Use the resulting models for the full WildFusion pipeline.



