from pyhdf.SD import SD
import os

filenameCC = os.path.join("CloudSatCALIPSO","2007001005141_03607_CS_2B-CLDCLASS-LIDAR_GRANULE_P1_R05_E02_F00.hdf")
filenameMOD = os.path.join("MODIS_raw","MAC021S0.A2007001.0000.002.2017117214630.hdf")

# Open the HDF4 file
hdf_file = SD(filenameCC, SD.READ)

# Explore the structure
print(hdf_file.info())

# Access dataset
dataset = hdf_file.select('dataset_name')

# Read data
data = dataset.get()

# Close the HDF4 file
hdf_file.end()