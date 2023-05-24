# Author: Nishit Patel - n.t.patel@student.utwente.nl/nishit99patel@gmail.com

# At this point, you should have downloaded the data using the sentinelsat -
# - API script and the data files should be stored in a 'data directory' in -
# - zip format.
# this script will allow the users to:
# 1. Sort the data: As you'd want to train your model with pairs of Sentinel-3 -
# - and Sentinel-1 images that have closest dates of acquisition.
# 2. Unzip the data: Once the data is sorted, you'd want to unzip the files.
# 3. Preprocess the data: Once the data is unzipped, you'd want to preprocess - 
# - both Sentinel-1 and Sentinel-3 data using 'snappy' package of ESA's SNAP.

# ----- imports -----
import os
import zipfile
import datetime
import shutil
import snappy
from snappy import ProductIO

# ----- sort the data -----
def sorter(data_dir):
    '''
    parameters -
    data_dir: input path to Sentinel-3 and Sentinel-1 zip data
    '''
    s3_date_list = []
    s1_date_dict = {}

    # list of zip files
    zip_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.zip')]

    # make subfolders based on dates
    for zip_file in zip_files:

        # extract date from s3 filename and make subfolders
        if 'S3A'in zip_file:
            s3_date = datetime.datetime.strptime(os.path.basename(zip_file)[16:24], '%Y%m%d')
            s3_date_list.append(s3_date)

            # create subdirectory with s3 dates
            s3_dir = os.path.join(data_dir, s3_date.strftime('%Y%m%d'))
            os.makedirs(s3_dir, exist_ok=True)

        elif 'S1A' in zip_file:
            s1_date = datetime.datetime.strptime(os.path.basename(zip_file)[17:25], '%Y%m%d')

            # create dictionary with s1 dates
            if s1_date not in s1_date_dict:
                s1_date_dict[s1_date] = zip_file

    # populate the subfolders with S3 data   
    for zip_file in zip_files:
        if 'S3A' in zip_file:
            src_date =  datetime.datetime.strptime(os.path.basename(zip_file)[16:24], '%Y%m%d')
            date_diff = [(src_date - date).days for date in date_list]
            nearest_index = min(range(len(date_diff)), key=lambda i:abs(date_diff[i]))
            nearest_date = date_list[nearest_index].strftime('%Y%m%d')
            for folder in os.listdir(data_dir):
                folder_path = os.path.join(data_dir, folder)
                # check if it is folder and not file
                if os.path.isdir(folder_path):
                    folder_name = os.path.basename(folder_path)
                    # check nearest date
                    if str(nearest_date) == folder_name:
                        shutil.copy(zip_file, folder_path)

    # populate the subfolders with S1 data
    for folder in os.listdir(data_dir):
        date_diff_list = []
        folder_path = os.path.join(data_dir, folder)
        # check if it is folder and not file
        if os.path.isdir(folder_path):
            folder_name = os.path.basename(folder_path)
            if '.ipynb' not in folder_name:
                folder_name_dt = datetime.datetime.strptime(folder_name, '%Y%m%d')
                for k, v in s1_date_dict.items():
                    diff = (folder_name_dt - k).days
                    date_diff_list.append(diff)
                nearest_index_s1 = min(range(len(date_diff_list)), key=lambda i:abs(date_diff_list[i]))
                shutil.copy(s1_date_dict[list(s1_date_dict)[nearest_index_s1]], folder_path)
                
    return 'files should be sorted!'

# sort data
data_dir = ''  # input path to the folder where you downloaded S3 and S1 data
sorter(data_dir=data_dir)

# ----- unzip the data -----
def unzipper(data_dir):
    '''
    parameters -
    data_dir: input path to Sentinel-3 and Sentinel-1 zip data
    '''
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith('.zip'):
                    zip_path = os.path.join(folder_path, file)
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(folder_path)
                    os.remove(zip_path)
                    
    return 'files should be unzipped!'

# unzip data
data_dir = ''  # input path to the folder where you downloaded S3 and S1 data
unzipper(data_dir=data_dir)

# ----- pre-processing S1 and S3 data -----
# sentinel-1 preprocessing function
def s1_preprocessing(input_data, roi, proj, output_name_linear, output_name_db):
    '''
    parameters -
    input_data = path to Sentinel-1 '.SAFE' file
    roi = area of interest (polygon)
    proj = crs projection system
    output_name_linear = name of the preprocessed output file (linear units)
    output_name_db = name of the preprocessed output file (db units)
    '''
    s1_preprocessed = []
    
    # read S1 product
    s1_data = ProductIO.readProduct(input_data)
    
    # susbet area
    WKTReader = snappy.jpy.get_type('org.locationtech.jts.io.WKTReader')
    geom = WKTReader().read('POLYGON((3.9084 51.8001,3.9084 52.0584,4.6239 52.0584,4.6239 51.8001,3.9084 51.8001))')

    parameters = snappy.HashMap()
    parameters.put('copyMetadata', True)
    parameters.put('geoRegion', roi)
    s1_subset = snappy.GPF.createProduct("Subset", parameters, s1_data)
    
    # apply orbit file
    parameters = snappy.HashMap()
    parameters.put("Apply-Orbit-File", True)
    orbit = snappy.GPF.createProduct("Apply-Orbit-File", parameters, s1_subset)
    
    # thermal noise removal
    parameters = snappy.HashMap()
    parameters.put("removeThermalNoise", True)
    thermal_noise = snappy.GPF.createProduct("ThermalNoiseRemoval", parameters, orbit)
    
    # radiometric calibration
    parameters = snappy.HashMap()
    parameters.put("outputSigmaBand", True)
    parameters.put("sourceBands", "Intensity_VH,Intensity_VV")
    parameters.put("selectedPolarisations", "VH,VV")
    parameters.put("outputImageScaleInDb", False)
    r_calibrated = snappy.GPF.createProduct("Calibration", parameters, thermal_noise)
    
    # speckle filtering
    parameters = snappy.HashMap()
    parameters.put("filter", "Lee")
    parameters.put("filterSizeX", 5)
    parameters.put("filterSizeY", 5)
    filtered = snappy.GPF.createProduct("Speckle-Filter", parameters, r_calibrated)
    
    # terrain correction
    parameters = snappy.HashMap()
    parameters.put("demName", "SRTM 3Sec")
    parameters.put("imgResamplingMethod", "BILINEAR_INTERPOLATION")
    parameters.put("pixelSpacingInMeter", "10")
    parameters.put("mapProjection", proj)
    parameters.put("saveProjectedLocalIncidenceAngle", False)
    parameters.put("saveSelectedSourceBand", True)
    t_corrected = snappy.GPF.createProduct("Terrain-Correction", parameters, filtered)
    
    # add product to list
    s1_preprocessed.append(t_corrected)
    
    # linear to db
    parameters = snappy.HashMap()
    parameters.put('sourceBands', 'Sigma0_VH,Sigma0_VV')
    db_converted = snappy.GPF.createProduct('LinearToFromdB', parameters, t_corrected)
    
    # add to list
    s1_preprocessed.append(db_converted)
    
    # save products
    ProductIO.writeProduct(t_corrected, output_name_linear, 'GeoTIFF')
    ProductIO.writeProduct(db_converted, output_name_db, 'GeoTIFF')
    
    # clear memory
    s1_data.dispose()
    t_corrected.dispose()
    db_converted.dispose()
    
    return s1_preprocessed

# sentinel-3 preprocessing function
def s3_preprocessing(input_data, roi, proj, output_name_s3):
    '''
    parameters -
    input_data = path to Sentinel-3 '.SEN3' file
    roi = area of interest (polygon)
    proj = crs projection system
    output_name_s3 = name of the preprocessed output file
    '''
    s3_preprocessed = []
    
    # read product
    s3_data = ProductIO.readProduct(input_data)
    
    # subset area
    parameters = snappy.HashMap()
    parameters.put('geoRegion', roi)
    s3_subset = snappy.GPF.createProduct("Subset", parameters, s3_data)
    
    # reproject the data
    parameters = snappy.HashMap()
    parameters.put('crs', proj)
    parameters.put('resampling', 'Bilinear')
    parameters.put('pixelSizeX', '1000')
    parameters.put('pixelSizeY', '1000')
    parameters.put('orthoRectify', False)
    s3_reproj_product = snappy.GPF.createProduct('Reproject', parameters, s3_subset)
    
    # spectral subset
    parameters = snappy.HashMap()
    parameters.put('sourceBands', 'LST,NDVI,fraction,bayes_in')
    s3_spectral_subset= snappy.GPF.createProduct('Subset', parameters, s3_reproj_product)
    
    # write tiff
    ProductIO.writeProduct(s3_spectral_subset, output_name_s3, 'GeoTIFF')
    
    # append product to list
    s3_preprocessed.append(s3_spectral_subset)
    
    # clear memory
    s3_data.dispose()
    s3_spectral_subset.dispose()
    
    return s3_preprocessed

# define projection
proj = '''PROJCS["WGS 84 / UTM zone 32N",
    GEOGCS["WGS 84",
        DATUM["WGS_1984",
            SPHEROID["WGS 84",6378137,298.257223563,
                AUTHORITY["EPSG","7030"]],
            AUTHORITY["EPSG","6326"]],
        PRIMEM["Greenwich",0,
            AUTHORITY["EPSG","8901"]],
        UNIT["degree",0.0174532925199433,
            AUTHORITY["EPSG","9122"]],
        AUTHORITY["EPSG","4326"]],
    PROJECTION["Transverse_Mercator"],
    PARAMETER["latitude_of_origin",0],
    PARAMETER["central_meridian",9],
    PARAMETER["scale_factor",0.9996],
    PARAMETER["false_easting",500000],
    PARAMETER["false_northing",0],
    UNIT["metre",1,
        AUTHORITY["EPSG","9001"]],
    AXIS["Easting",EAST],
    AXIS["Northing",NORTH],
    AUTHORITY["EPSG","32632"]]'''

# define roi for Rotterdam
WKTReader = snappy.jpy.get_type('org.locationtech.jts.io.WKTReader')
roi = WKTReader().read('POLYGON((3.9084 51.8001,3.9084 52.0584,4.6239 52.0584,4.6239 51.8001,3.9084 51.8001))')

# run the preprocessing functions iteratively to preprocess all the downloaded data
data_dir = ''  # path to directory where all your data is stored
s1_filter_list = []
s1_linear_mov_list = []
s1_db_mov_list = []

for folder in os.listdir(data_dir):
    folder_path = os.path.join(data_dir, folder)
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            if file.endswith('.SAFE'):
                if file not in s1_filter_list:
                    s1_filter_list.append(file)
                    s1_input_data = os.path.join(folder_path, file)
                    output_name_linear = s1_input_data[:26] + 's1_'+ s1_input_data[43:51] + '_linear' + '.tif' 
                    output_name_db = s1_input_data[:26] + 's1_'+ s1_input_data[43:51] + '_db' + '.tif'
                    s1_preprocessing(s1_input_data, roi, proj, output_name_linear, output_name_db)
                    # add to moving list to not preprocess same s1 data twice
                    s1_linear_mov_list.append(output_name_linear)
                    s1_db_mov_list.append(output_name_db)
                else: 
                    shutil.copy(s1_linear_mov_list[-1], folder_path)
                    shutil.copy(s1_db_mov_list[-1], folder_path)
            elif file.endswith('.SEN3'):
                s3_input_data = os.path.join(folder_path, file)
                output_name_s3 = s3_input_data[:26] + 's3_'+ s3_input_data[42:50] + '.tif'
                s3_preprocessing(s3_input_data, roi, proj, output_name_s3)
                
                
# at the end of this script, you should have preprocessed sentinel-1 and sentinel-3 '.tif'
# files saved to your dated sub-folders within the data directory
