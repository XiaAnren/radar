# Radar Data Processing

## Script Overview

This directory contains Python scripts for radar data processing and visualization. The scripts are organized by functionality as follows:

### Data Processing
- `standardization.py` - Standardizes single-site radar base data
- `milab_cappi.py` - Computes single-site CAPPI data using MeteoInfoLab
- `mosaic.py` - Creates radar mosaics by combining multiple station data
- `run.py` - Main pipeline script that sequentially calls standardization, CAPPI computation and mosaicking
- `fusion.py` - Fuses radar mosaics with SWAN data

### Radar Visualization
- `individual.py` - Visualizes single-site radar base data
- `swan.py` - Compares radar mosaics with SWAN data
- `plot.py` - Visualizes fused radar data products

### WRF Experiment Visualization
- `domain.py` - Visualizes WRF experiment domain configurations
- `case.py` - Visualizes WRF experiment results
- `sequence.py` - Creates time series plots of WRF experiment metrics

### Utilities & Configuration
- `build.sh` - Build script for setting up the data folder containing radar data
- `structure.py` - Stores and processes radar structure data
- `viztools.py` - Visualization utilities and helper functions
- `viztools.yaml` - Configuration settings for visualization tools
- `metrics.py` - Contains calculation classes for metrics like FSS

## Directory Structure
- `data/` - Contains input and output radar data files
- `images/` - Storage for generated visualizations and plots

## Processing Pipeline
The standard processing workflow is:
1. Standardize single-site radar base data (`standardization.py`)
2. Compute single-site CAPPI data (`milab_cappi.py`)
3. Create radar mosaics from multiple stations (`mosaic.py`)
4. Fuse radar mosaics with SWAN data (`fusion.py`)

This pipeline can be executed sequentially using:
```bash
python run.py
python fusion.py
```

## Visualization
- **Radar-specific visualizations**: `individual.py`, `swan.py`, `plot.py`
- **WRF experiment visualizations**: `domain.py`, `case.py`, `sequence.py`

---

*Last updated: 2025-08-30*  
*Note: Refer to individual script docstrings for detailed usage instructions and configuration options*