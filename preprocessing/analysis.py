import os
from osgeo import gdal
import numpy as np


filenameCC = os.path.join("CloudSatCALIPSO","2007001005141_03607_CS_2B-CLDCLASS-LIDAR_GRANULE_P1_R05_E02_F00.hdf")
filenameMOD = os.path.join("MODIS_raw","MAC021S0.A2007001.0000.002.2017117214630.hdf")


file = gdal.Open(filenameCC)

sds_list = file.GetSubDatasets()

for sds in sds_list:
    print(sds[1]) # i.e. the desc

print(np.shape(sds_list))

subdataset = gdal.Open(sds_list[1][0])

data_array = subdataset.ReadAsArray()

print(np.shape(data_array))

# Get metadata
metadata = file.GetMetadata()
