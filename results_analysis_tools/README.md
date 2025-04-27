# About
This directory encapsulates the logic for all analysis data processing generated from the 
tools defined in `evaluation_tools/`. The tools in this directory serves for processing the 
data to yield the experimental results ready to use for the thesis typically in a form of 
LaTeX table or Figure.

# Results Analyzer
One of the core interfaces of this directory is defined in `result_analyzer.py` that encapsulates
variety of plugins that serve for processing the data to format ready for either plotting or
to be inserted to LaTeX document (in terms of tables). 

For example of usage this tool please refer to `analysis_overview.ipynb` where the overview of all
functionalities of the tool is outlined including plotting the results using `results_plotter.py`
another interface of this directory encapsulating all plotting logic.

# Directory Content
This directory contains:

- `analysis_overview.ipynb` - Jupyter notebook that outlines the workflow through all functionalities of the tools defined in this directory (it also outlines process of generating all plots and tables in our thesis)
- `result_analyzer.py` - main script encapsulating the logic processing the analysis results
- `results_plotter.py` - interface for plotting the processed results
- `plugins/` - Directory containing definition of all plugins used for analysis results processing.
- `fields/` - All common namings, types and fields through this tool.
- `plotting/` - Directory containing all plotting functions.