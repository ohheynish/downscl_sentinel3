# Author: Nishit Patel - n.t.patel@student.utwente.nl/nishit99patel@gmail.com

# ----- imports -----
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
import folium
from folium.plugins import Draw
from shapely.geometry import Polygon
from datetime import datetime, timedelta

# ----- draw area of interest -----
# the below script will generate a geojson file for the drawn area of interest
m = folium.Map(location=[51.921343, 4.294729], zoom_start=10)
Draw(
    export=True,
    filename="rotterdam.geojson",
    show_geometry_on_click = True
).add_to(m)
m

# ----- access API -----
username = ''  # your username on scihub.copernicus.eu
password = ''  # your password on scihub.copernicus.eu
api = SentinelAPI(username, password, 'https://scihub.copernicus.eu/dhus')

# ----- define area of interest -----
geojson_path = ''  # path to your geojson file
area_of_interest = geojson_to_wkt(read_geojson(geojson_path))

# ----- query Sentinel-1 data -----
# here, we are only interested in 'Sentinel-1-A' products as 'Sentinel-1-B' -
# - became unoperational sometime in December 2021
# further, we are only interested in 'GRD' (ground-range detected) products -
# - as we don't need any phase information
# the orbit direction is set to 'Ascending'
# time of acquisition for 'Ascending' orbit: ~17:30
# time of acquisition for 'Descending' orbit: ~4:30
# the time frame (beginposition) is entire month of August 2022
sentinel_1_data = api.query(
                            area_of_interest,
                            beginposition='[2022-08-01T00:00:00.000Z TO 2022-08-31T00:00:00.000Z]',
                            platformname='Sentinel-1',
                            producttype='GRD',
                            filename='S1A_IW*',
                            orbitdirection='ASCENDING'
                           )

# save the above queried products to a dataframe to facilitate filtering -
# - and batch download
sentinel_1_df = api.to_dataframe(sentinel_1_data)

# ----- query Sentinel-3 data -----
# here, we are only interested in 'Sentinel-3-A' products
# further, we are only interested in 'SL_2_LST' (Land Surface Temperature Level-2)
# the orbit direction is set to 'Descending'
# time of acquisition for 'Ascending' orbit: ~20:30
# time of acquisition for 'Descending' orbit: ~10:30
# the time frame (beginposition) is entire month of August 2022
sentinel_3_data = api.query(
                            area_of_interest,
                            beginposition='[2022-08-01T00:00:00.000Z TO 2022-08-31T00:00:00.000Z]',
                            platformname='Sentinel-3',
                            producttype='SL_2_LST___',
                            filename='S3A*',
                            orbitdirection='DESCENDING'
                           )

# save the above queried products to a dataframe to facilitate filtering -
# - and batch download
sentinel_3_df = api.to_dataframe(sentinel_3_data)

# ----- download products -----
# if you are downloading previous year data, you'd have to wait a while -
# - as old data is stored in 'long term archive' on 'scihub.copernicus.eu'
data_dir = ''  # path to the directory you want to save your data to
api.download_all(sentinel_3_df.index, directory_path=data_dir, checksum=False)
api.download_all(sentinel_1_df.index, directory_path=data_dir, checksum=False)
