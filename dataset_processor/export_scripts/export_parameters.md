# Sheets
Basic information about the layer data from raw SNN dataset.

Variants:

```python
['V1_Exc_L2/3', 'V1_Inh_L2/3', 'V1_Exc_L4', 'V1_Inh_L4', 'X_ON' 'X_OFF']
```

```python
sheets = ['V1_Exc_L2/3', 'V1_Inh_L2/3', 'V1_Exc_L4', 'V1_Inh_L4', 'X_ON' 'X_OFF']
sheet_folders  = ['V1_Exc_L23', 'V1_Inh_L23', 'V1_Exc_L4', 'V1_Inh_L4', 'X_ON' 'X_OFF']
```



# Paths
Paths to raw SNN dataset.

## Layer Directories

### L2/3:
```bash
/CSNG/baroni/mozaik-models/LSV1M/20240117-111742[param_nat_img.defaults]CombinationParamSearch{trial:[0],baseline:500}/
```

### L4 and LGN:
```bash
/CSNG/baroni/mozaik-models/LSV1M/20240124-093921[param_nat_img.defaults]CombinationParamSearch{trial:[0],baseline:500}/
```

### Test dataset:
```bash
/CSNG/baroni/mozaik-models/LSV1M/20240911-181115[param_nat_img.defaults]CombinationParamSearch{trial:[0],baseline:20}/
```

Note: The examples `300000-300050` and `300150-300200` are corrupted and cannot
be currently used for the extraction.

## Experiment Subdirectories

In format:
```bash
NewDataset_Images_from_{start}_to_{end}_ParameterSearch_____baseline:{start}_trial:0", 
```

Example:
```bash
NewDataset_Images_from_50000_to_50100_ParameterSearch_____baseline:50000_trial:0", 
```

