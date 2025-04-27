# About
This directory entails all tools used prevailingly for the raw evaluation results processing and 
additionally the analysis of the dataset. The results of the analyses are expected to be further 
processed by the tools from `results_analysis_tools` where the final evaluation for plotting the
results and retrieving statistic analysis results should be computed.

# Evaluation Processor
Main script covering whole functionality of this directory is `evaluation_processor.py` that serves 
as a user interface for running several analyses. The tool covers several functionalities that are
implemented as plugins in the directory `plugins/`. For further description of the specific plugin
or the evaluation processor please refer to the source code where the comprehensive documentation of
each tool is documented and defined.

The most important for the evaluation processor analysis is the selection of the argument `--action`
that specifies the type of the analysis that should be done. Additional arguments are selected 
specifically based on the action type. 

These actions are:
- `full_dataset` - Analysis of the full dataset using different time bins.
- `subset_dataset` - Analysis of the dataset using different model subsets.
- `wandb_analysis` - Analysis of the Weights and Biases results.
- `prediction_analysis` - Analysis of the raw evaluation predictions right from the model.

Additionally for more information regarding usage of the evaluation processor one may run the command:

```bash
python evaluation_processor.py --help
```

# Directory Structure
The directory contains:
- `evaluation_processor.py` - Main script that serves as a user interface and entails all functionality covered in this directory.  
- `evaluation_results/` - Directory where all analysis results of the functionality from this directory and where also the evaluation results of the model should be stored.
- `plugins/` - All tool plugins serving different types of analyses.
- `fields/` - All specific fields, types and namings used in the tool.
- `scripts/` - Additional support scripts for functionality out of the tool range.
- `wintermute_jobs/` - Scripts to run the evaluation on Wintermute cluster.
