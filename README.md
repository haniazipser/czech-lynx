# czech-lynx

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

## Start training

```bash
# New run
python run_baseline.py 
```


## Visualizing dataset

```bash
python explore_data.py 
```


