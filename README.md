# downscl_sentinel3
 This repo is dedicated to downscaling Sentinel-3 SLSTR LST using RVI (radar vegetation index) generated from Sentinel-1 SAR,
 ESA WorldCover, and DSM data
 
 The 'sentinelsat.py' script enables users to download the corresponding Sentinel-3 and 
 Sentinel-1 data from European Space Agency's (ESA) https://scihub.copernicus.eu/ database
 
 Once the data is downloaded, the 'preprocessing.py' script preprocesses both Sentinel-1
 GRD SAR and Sentinel-3 SLSTR LST data. The main processes included in this script are:
 'data sorting and pairing' and 'preprocessing (terrain correction, thermal noise removal,
 coregistration between Sentinel-1 and Sentinel-3, ...)
 
 Finally, the 'model.py' script collects predictors and target variables for the
 downscaling model from the preprocessed data. Here, the downscaling algorithm utilizes
 Random Forest regression as the backbone model
