# Run Wintermute

```bash
srun --cpus-per-task=64 --pty bash
```


# Sheets

```bash
--sheet=
```

Variants:

```python
['V1_Exc_L2/3', 'V1_Inh_L2/3', 'V1_Exc_L4', 'V1_Inh_L4', 'X_ON' 'X_OFF']
```

```python
sheets = ['V1_Exc_L2/3', 'V1_Inh_L2/3', 'V1_Exc_L4', 'V1_Inh_L4', 'X_ON' 'X_OFF']
sheet_folders  = ['V1_Exc_L23', 'V1_Inh_L23', 'V1_Exc_L4', 'V1_Inh_L4', 'X_ON' 'X_OFF']
```



# Paths

## Layer Directories
For L2/3:

```bash
/CSNG/baroni/mozaik-models/LSV1M/20240117-111742[param_nat_img.defaults]CombinationParamSearch{trial:[0],baseline:500}/
```

For L4 or LGN:

```bash
/CSNG/baroni/mozaik-models/LSV1M/20240124-093921[param_nat_img.defaults]CombinationParamSearch{trial:[0],baseline:500}/
```

## Experiment Subdirectories

In format:
```bash
NewDataset_Images_from_{start}_to_{end}_ParameterSearch_____baseline:{start}_trial:0", 
```

Example:
```bash
NewDataset_Images_from_50000_to_50100_ParameterSearch_____baseline:50000_trial:0", 
```

