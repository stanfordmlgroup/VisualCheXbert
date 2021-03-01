# VisualCheXbert: Addressing the Discrepancy Between Radiology Report Labels and Image Labels

## Prerequisites 

Create conda environment

```
conda env create -f environment.yml
```

Activate environment

```
conda activate visualCheXbert
```

By default, all available GPU's will be used for labeling in parallel. If there is no GPU, the CPU is used. You can control which GPU's are used by appropriately setting CUDA_VISIBLE_DEVICES. The batch size by default is 18 but can be changed inside constants.py

## Checkpoint download

Download our trained model checkpoints here: https://drive.google.com/file/d/1Od62LAbdmfcK6W8TfzNcLIUouxWZoMHn/view?usp=sharing.

## Usage

### Label reports with CheXbert

Put all reports in a csv file under the column name "Report Impression". Let the path to this csv be {path to reports}. Download and unzip the checkpoint folder, and let the path to it be {path to checkpoint folder}. Let the path to your desired output folder by {path to output dir}. 

```
python label.py -d={path to reports} -o={path to output dir} -c={path to checkpoint folder} 
```

The output file with labeled reports is {path to output dir}/labeled_reports.csv

Run the following for descriptions of all command line arguments:

```
python label.py -h
```

**Ignore any error messages about the size of the report exceeding 512 tokens. All reports are automatically cut off at 512 tokens.**
