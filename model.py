# Author: Nishit Patel - n.t.patel@student.utwente.nl/nishit99patel@gmail.com

# At this point, you'd have preprocessed Sentinel-1 and Sentinel-3 data which -
# - are now ready to be used as inputs for downscaling. However, first we -
# - create RVI (radar vegetation index) which will also be used as a model input.

# ----- imports -----
from osgeo import gdal
from osgeo import gdalconst
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# ----- Radar Vegetation Index (RVI) -----
# based on the work of Milad Mahour - m.mahour@rotterdam.nl
# formula: 0.5*((vv - 4*vh)/(vv + 4*vh) + 1), where vv and vh are in linear units

# rvi function
def create_rvi(s1_data, s3_data, rvi_output_path):
    '''
    parameters - 
    s1_data: path to preprocessed sentinel-1 '.tif' file (linear units)
    s3_data: path to preprocessed sentinel-3 '.tif' file
    rvi_output_path: path alongwith filename of the output rvi '.tif'
    '''
    s3_data_extent_corr = gdal.Open(s3_data)
    s1_data_linear = gdal.Open(s1_data)
    
    # extent correction
    geoTransform = s3_data_extent_corr.GetGeoTransform()
    ulx = geoTransform[0]
    uly = geoTransform[3]
    lrx = ulx + geoTransform[1] * s3_data_extent_corr.RasterXSize
    lry = uly + geoTransform[5] * s3_data_extent_corr.RasterYSize
    extent = [ulx, uly, lrx, lry]
    
    s1_data_linear_extent = gdal.Translate('/vsimem/in_memory_output_s1_linear.tif',
                                           s1_data_linear, projWin=extent,
                                           outputType=gdalconst.GDT_Float32, noData=np.nan)
    
    vh_linear_arr, vv_linear_arr = s1_data_linear_extent.ReadAsArray()[0], s1_data_linear_extent.ReadAsArray()[1]
    
    # RVI milad
    vv_linear_arr[vv_linear_arr > 1] = 1
    vh_linear_arr[vh_linear_arr > 1] = 1

    vv_linear_arr[vv_linear_arr < 0] = 0
    vh_linear_arr[vh_linear_arr < 0] = 0
    
    # formula
    r1 = vv_linear_arr-4*vh_linear_arr
    r2 = vv_linear_arr+4*vh_linear_arr
    ratio = r1/r2

    rvi_milad = 0.5*(ratio+1)
    
    # save rvi as tiff
    driver = gdal.GetDriverByName('GTiff')
    new_ds = driver.Create(rvi_output_path,
                           s1_data_linear_extent.RasterXSize, s1_data_linear_extent.RasterYSize,
                           1, gdal.GDT_Float32)
    new_ds.SetGeoTransform(s1_data_linear_extent.GetGeoTransform())
    new_ds.SetProjection(s1_data_linear_extent.GetProjection())
    new_ds.GetRasterBand(1).SetNoDataValue(np.nan)
    new_ds.GetRasterBand(1).WriteArray(rvi_milad)
    new_ds = None
    
    return 'rvi generated!'

# run the function for all data
data_dir = ''  # path to your main data folder (not the subfolders)
for folder in os.listdir(data_dir):
    folder_path = os.path.join(data_dir, folder)
    if os.path.isdir(folder_path) and '.ipynb' not in folder_path:
        for file in os.listdir(folder_path):
            if 'linear' in file:
                s1_data = os.path.join(folder_path, file)
                rvi_output_path = folder_path + '/rvi_' + folder_path[-8:] +'.tif'
            elif file.endswith('.tif') and 's3' in file:
                s3_data = os.path.join(folder_path, file)
            else:
                continue
        create_rvi(s1_data=s1_data, s3_data=s3_data, rvi_output_path=rvi_output_path)
        
# ----- downscaling models -----
# once we have rvi generated, we are ready to train our models and estimate LST
# at 10 m.
# here we use Random Forest as our backbone model

# function for the whole model framework
def model(s3_data, s1_data, rvi_data, dsm_data, lulc_data, output_path, model_ver):
    '''
    parameters -
    s3_data: path to the preprocessed sentinel-3 '.tif' file
    s1_data: path to the preprocessed sentinel-1 '.tif' file (db units)
    rvi_data: path to the radar vegetation index '.tif. file
    dsm_data: path to digital surface model (dsm) '.tif' file
    lulc_data: path to ESA land-use land-cover (lulc) '.tif' file
    output_path: path to save the output downscaled LST
    model_ver: 'model_A' - random forest (rvi, dsm, lulc)
               'model_B' - random forest (vv, rvi, dsm, lulc)
    '''   
    
    downsc_output_products = []
    
    # read data
    s1_data = gdal.Open(s1_data)
    dsm_data = gdal.Open(dsm_data)
    lulc_data = gdal.Open(lulc_data)
    rvi_data = gdal.Open(rvi_data)
    s3_data = gdal.Open(s3_data)
    
    # extent correction
    geoTransform = s3_data.GetGeoTransform()
    ulx = geoTransform[0]
    uly = geoTransform[3]
    lrx = ulx + geoTransform[1] * s3_data.RasterXSize
    lry = uly + geoTransform[5] * s3_data.RasterYSize
    extent = [ulx, uly, lrx, lry]


    s1_data_extent = gdal.Translate('/vsimem/in_memory_output_dsm.tif',
                                     s1_data, projWin=extent, noData=np.nan)

    dsm_data_extent = gdal.Translate('/vsimem/in_memory_output_dsm.tif',
                                     dsm_data, projWin=extent, noData=np.nan)

    lulc_data_extent = gdal.Translate('/vsimem/in_memory_output_lulc.tif',
                                      lulc_data, projWin=extent, noData=np.nan)

    rvi_data_extent = gdal.Translate('/vsimem/in_memory_output_s1_linear.tif',
                                     rvi_data, projWin=extent, noData=np.nan)
    
    # read as arrays
    vh_arr, vv_arr = s1_data_extent.ReadAsArray()[0], s1_data_extent.ReadAsArray()[1]
    
    dsm_arr = dsm_data_extent.ReadAsArray()
    dsm_arr[dsm_arr > 1000] = np.nan

    lulc_arr = lulc_data_extent.ReadAsArray()
    lulc_arr = lulc_arr.astype('float')
    lulc_arr[lulc_arr <= 0] = np.nan

    rvi_arr = rvi_data_extent.ReadAsArray()

    lst_arr = s3_data.ReadAsArray()[2]
    lst_arr[lst_arr <= 0] = np.nan
    
    # upscaling (Cubic reduce both lst s3 and sar s1)
    s1_1000_data = gdal.Warp('/vsimem/in_memory_output_s1_upscaled.tif',
                               s1_data_extent, xRes=1000, yRes=1000,
                               resampleAlg = gdal.GRA_Cubic)
    vh_1000, vv_1000 = s1_1000_data.ReadAsArray()[0], s1_1000_data.ReadAsArray()[1]
    
    dsm_1000_data = gdal.Warp('/vsimem/in_memory_output_dsm_upscaled.tif',
                               dsm_data_extent, xRes=1000, yRes=1000,
                               resampleAlg = gdal.GRA_Cubic)
    dsm_1000 = dsm_1000_data.ReadAsArray()

    lulc_1000_data = gdal.Warp('/vsimem/in_memory_output_dsm_upscaled.tif',
                               lulc_data_extent, xRes=1000, yRes=1000,
                               resampleAlg = gdal.GRA_Mode)
    lulc_1000 = lulc_1000_data.ReadAsArray().astype('float')
    lulc_1000[lulc_1000 <= 0] = np.nan
    
    rvi_1000_data = gdal.Warp('/vsimem/in_memory_output_rvi_upscaled.tif', 
                               rvi_data_extent, xRes=1000, yRes=1000,
                               resampleAlg = gdal.GRA_Cubic)
    rvi_1000 = rvi_1000_data.ReadAsArray()

    s3_1000_data = gdal.Warp('/vsimem/in_memory_output_s3_upscaled.tif', s3_data, xRes=1000, yRes=1000,
                            resampleAlg = gdal.GRA_Cubic)
    lst_1000 = s3_1000_data.ReadAsArray()[2]
    lst_1000[lst_1000 <= 0] = np.nan
    
    clouds_1000 = s3_1000_data.ReadAsArray()[3]
    clouds_1000[clouds_1000 < 0] = np.nan
    
    # save arrays to dataframes
    data_df_1000 = pd.DataFrame(lst_1000.flatten().T, columns=['lst'])
    data_df_1000['clouds'] = clouds_1000.flatten().T
    data_df_1000['vv'] = vv_1000.flatten().T
    data_df_1000['dsm'] = dsm_1000.flatten().T
    data_df_1000['lulc'] = lulc_1000.flatten().T
    data_df_1000['rvi_m'] = rvi_1000.flatten().T

    # drop NaN rows
    f_data_df_1000 = data_df_1000.dropna()
    
    # drop cloudy pixels
    fc_data_df_1000 = f_data_df_1000[f_data_df_1000.clouds != 2]

    if not f_data_df_1000.empty:
        # calculate proportion of cloudy pixels in Rotterdam
        prop_clouds = 1 - (len(fc_data_df_1000)/len(f_data_df_1000))
        print(prop_clouds)
        # select threshold
        th = 0.3  # 30% of area covered in clouds

        if prop_clouds <= th:
            if model_ver == 'model_A':
                # save data as predictors and targets for training
                predictors = fc_data_df_1000.drop(['lst', 'clouds', 'vv'], axis=1)
                target = fc_data_df_1000['lst']

            elif model_ver == 'model_B':
                # save data as predictors and targets for training
                predictors = fc_data_df_1000.drop(['lst', 'clouds'], axis=1)
                target = fc_data_df_1000['lst']

            # train-test
            x_train, x_test, y_train, y_test = train_test_split(predictors, target,
                                                                test_size = 0.2,
                                                                random_state = 17)

            # create a grid with hyperparameters for model tuning
            n_estimators = [int(x) for x in np.linspace(start=10, stop=1000, num=100)]
            max_depth = [int(x) for x in np.linspace(1, 50, num=25)]
            max_depth.append(None)

            random_grid = {'n_estimators': n_estimators,
                           'max_depth': max_depth}

            # model
            model = RandomForestRegressor()

            rf_random = RandomizedSearchCV(estimator=model,
                                           param_distributions=random_grid,
                                           n_iter=100,
                                           cv=3,
                                           scoring='neg_root_mean_squared_error',
                                           return_train_score=True,
                                           random_state=21,
                                           n_jobs=-1)
            rf_random.fit(x_train, y_train)

            # get the best model
            tuned_model = rf_random.best_estimator_

            # predict on entire image at 1000 m 
            lst_pred_1000 = tuned_model.predict(predictors)

            # bring everything back to original dataframe
            n_f_data_df_1000 = fc_data_df_1000.copy()
            n_f_data_df_1000['lst_pred_1000'] = lst_pred_1000
            n_f_data_df_1000 = n_f_data_df_1000['lst_pred_1000']
            final_data_df_1000 = data_df_1000.join(n_f_data_df_1000)

            # prepare residuals
            pred_arr_1000 = final_data_df_1000['lst_pred_1000'].values.reshape(-1, lst_1000.shape[1])
            residual_arr = lst_arr - pred_arr_1000

            # upscaling to 10 m (cubic reduction)
            s1_10_data = gdal.Warp('/vsimem/in_memory_output_s1_10_resampled.tif', 
                                     s1_data_extent, xRes=10, yRes=10,
                                     resampleAlg = gdal.GRA_Cubic) # enlarge window size
            vh_10, vv_10 = s1_10_data.ReadAsArray()[0], s1_10_data.ReadAsArray()[1]

            dsm_10_data = gdal.Warp('/vsimem/in_memory_output_dsm_10_resampled.tif', 
                                     dsm_data_extent, xRes=10, yRes=10,
                                     resampleAlg = gdal.GRA_Cubic)
            dsm_10 = dsm_10_data.ReadAsArray()

            lulc_10_data = gdal.Warp('/vsimem/in_memory_output_dsm_upscaled.tif',
                                       lulc_data_extent, xRes=10, yRes=10,
                                       resampleAlg = gdal.GRA_Mode)
            lulc_10 = lulc_10_data.ReadAsArray().astype('float')
            lulc_10[lulc_10 <= 0] = np.nan

            rvi_10_data = gdal.Warp('/vsimem/in_memory_output_rvi_upscaled.tif', 
                                       rvi_data_extent, xRes=10, yRes=10,
                                       resampleAlg = gdal.GRA_Cubic)
            rvi_10 = rvi_10_data.ReadAsArray()

            # save as dataframe
            data_df_10 = pd.DataFrame(vv_10.flatten().T, columns=['vv'])
            data_df_10['dsm'] = dsm_10.flatten().T
            data_df_10['lulc'] = lulc_10.flatten().T
            data_df_10['rvi_m'] = rvi_10.flatten().T

            # drop NaN
            f_data_df_10 = data_df_10.dropna()

            if model_ver == 'model_A':
                predictors_10 = f_data_df_10.drop(['vv'], axis=1)

            elif model_ver == 'model_B':
                predictors_10 = f_data_df_10

            lst_pred_10 = tuned_model.predict(predictors_10)

            # bring back to original 10 m dataframe
            n_f_data_df_10 = f_data_df_10.copy()
            n_f_data_df_10['lst_pred_10'] = lst_pred_10
            n_f_data_df_10 =  n_f_data_df_10[['lst_pred_10']]
            final_data_df_10 = data_df_10.join(n_f_data_df_10)

            # residual correction
            residual_arr_10 = residual_arr.repeat(100, 0).repeat(100, 1)
            final_data_df_10['residuals'] = residual_arr_10.flatten().T
            final_data_df_10['lst_pred_res_10'] = final_data_df_10['lst_pred_10'] + final_data_df_10['residuals']

            # save dataframe columns to arrays
            lst_pred_10_2d = final_data_df_10.lst_pred_10.values.reshape(-1, rvi_10.shape[1])
            lst_pred_res_10_2d = final_data_df_10.lst_pred_res_10.values.reshape(-1, rvi_10.shape[1])

            # convert to celsius
            lst_pred_10_2d_c = lst_pred_10_2d - 273.15
            lst_pred_res_10_2d_c = lst_pred_res_10_2d - 273.15

            # add products to a list
            downsc_output_products.append(lst_pred_10_2d_c)
            downsc_output_products.append(lst_pred_res_10_2d_c)

            # save as tiffs
            for i, product in enumerate(downsc_output_products):
                if i == 0:
                    output_name = output_path + '/dlst_' + output_path[-8:] + '_' + model_ver + '.tif'
                else:
                    output_name = output_path + '/dlst_res_' + output_path[-8:] + '_' + model_ver + '.tif'
                driver = gdal.GetDriverByName('GTiff')
                new_ds = driver.Create(output_name, 
                                       rvi_data_extent.RasterXSize, rvi_data_extent.RasterYSize,
                                       1, gdal.GDT_Float32)
                new_ds.SetGeoTransform(rvi_data_extent.GetGeoTransform())
                new_ds.SetProjection(rvi_data_extent.GetProjection())
                new_ds.GetRasterBand(1).SetNoDataValue(np.nan)
                new_ds.GetRasterBand(1).WriteArray(product)
                new_ds = None
                print('done')

        else:
            print('the input images have bad quality data')
            
    else:
        print('no values found! - check input images')
    return 'model output in folder!'

# run the model for all the data
data_dir = ''  # path to your main data folder (not the subfolders)
for folder in os.listdir(data_dir):
    folder_path = os.path.join(data_dir, folder)
    if os.path.isdir(folder_path) and '.ipynb' not in folder_path:
        for file in os.listdir(folder_path):
            if 's3' in file and file.endswith('.tif'):
                s3_data = os.path.join(folder_path, file)
            elif 'db' in file and file.endswith('.tif'):
                s1_data = os.path.join(folder_path, file)
            elif 'rvi' in file:
                rvi_data = os.path.join(folder_path, file)
            elif 'dsm' in file:
                dsm_data = os.path.join(folder_path, file)
            elif 'lulc' in file:
                lulc_data = os.path.join(folder_path, file)
        print(s3_data, rvi_data, dsm_data, lulc_data)
        model(s3_data=s3_data, s1_data=s1_data, rvi_data=rvi_data, dsm_data=dsm_data,
                  lulc_data=lulc_data, output_path=folder_path, model_ver='model_A')
        print()
        
# at the end of this script, you should have two downscaled LST '.tif' files
# (without residual correction and with residual correction) in your dated subfolders
