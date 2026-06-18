# Anemoi Case Study
This repository contains scripts, notebooks and files to evaluate the forecasts for the Donnie Creek wildfire. 

**Useful Links** 

Link to Google Drive resource: https://drive.google.com/drive/u/0/folders/1iqwsFyJ6j5r8hkQA8rcx-EowqtlSDpkb. \
Link to Verif data format: https://github.com/WFRT/verif/wiki/Arranging-my-own-data.

## Directory Structure
**Point Verification**
* `wrf2verif.py` extract WRF point forecasts into Verif-format NetCDF.
* `run_wrf2verif.sh` extract multiple WRF forecast files from multiple WRF forecast folders into Verif-format NetCDF.
* `extract_obs.py` extract BC Wildfire weather stations data in CSV format.
* `fill_obs.py` fill observation values from CSV into Verif ready NetCDF.
* `slice_wrf.ipynb` manually slice WRF forecast NetCDF file for sanity check.
* `observation.ipynb` manually extract and clean the BC Wildfire weather stations observation data.
* `data/` output files from `run_wrf2verif.sh`, `extract_obs.py`, `fill_obs.py`.

**2D Verification**
*  `satellite.ipynb` compare satellite observations, tifs with models forecasts.

## Getting Started
The case study is part of the evaluation of machine learning based forecasts produced by a "stretched-grid" model. \
Anemoi is an open-source platform used for machine learning weather forecasting models.

**Prerequisites**

**Environment Variables**

## Methodology

**Important time periods (UTC)**
* Ignition: 2023-05-13
* First run: 2023-05-13 ~ 2023-05-20 (worsen on 2023-05-15)
* Second run: 2023-05-21 ~ 2023-06-02 (not as severe as the other runs)
* Third run: 2023-06-04 ~ 2023-06-13 (worsen on 2023-06-07)
* Last significant fire: 2023-06-18

**Variables**
* Surface temperature 
* Fire and thermal anomalies (derived from brightness temperature)
* Relative humidity
* Water vapour
* Wind vector

## Observation Data

**BC Wildfire stations**: https://www.for.gov.bc.ca/ftp/HPR/external/!publish/BCWS_DATA_MART/2023/ 

Stations used: 
| ID  | Name          | Lat   | Lon      | Alt (m)   |
|-----|---------------|-------|----------|-----------|
| 129 | Pink Mountain | 56.94 | -122.70  | 960.10    |
| 120 | Silver        | 57.37 | -121.41  | 835.00    |
| 131 | Muskwa        | 57.88 | -123.62  | 769.00    |

**Satellite imagery**: https://drive.google.com/drive/folders/1oRro8_L23_qQ4Xyd_UUPo40_4Q-Zt_dE

## Forecast Data
* WRF: nextcloud:~/Share_Forecasts/WAC00WG-01
* Anemoi

**Author**

Jessie Chen\
&emsp;Assisted by Claude by Anthropic (claude-sonnet-4-6, claude.ai) for coding and documentation.
