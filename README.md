# Anemoi Case Study
This repository contains scripts, notebooks and files to evaluate the forecasts for the Donnie Creek wildfire.

**Useful links**

Link to Google Drive resource: https://drive.google.com/drive/u/0/folders/1iqwsFyJ6j5r8hkQA8rcx-EowqtlSDpkb \
Link to Verif data format: https://github.com/WFRT/verif/wiki/Arranging-my-own-data

**Authors**

Jessie Chen\
&emsp;Assisted by Claude by Anthropic (claude-sonnet-4-6, claude.ai) for coding and documentation

Cosmo Pearson-Young

**Directory**
* `wrf2verif.py` extract WRF point forecasts into Verif-format NetCDF
* `run_wrf2verif.sh` extract multiple WRF forecast files from multiple WRF forecast folders into Verif-format NetCDF
* `extract_obs.py` extract BC Wildfire weather stations data in CSV format.
* `fill_obs.py` fill observation values from CSV into Verif ready NetCDF
