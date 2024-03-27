from pyhdf.SD import SD, SDC
from pyhdf.HDF import HDF, HC
from pyhdf.VS import *
import os
import numpy as np
from scipy.spatial import cKDTree
import csv

def build_tree(lat, lon):
    return cKDTree(np.column_stack((lat, lon)))

def co_locate(tree, lat1, lon1, lat2, lon2, threshold):
    co_located_indices = []
    
    for i, (lat1, lon1) in enumerate(zip(lat2, lon2)):
        dist, idx = tree.query([lat1, lon1])
        if dist <= threshold:
            co_located_indices.append((idx, i))
            
    return co_located_indices

def calcRadiance(data):
    emissive = data[:]
    # Get dataset attributes
    attrs = data.attributes()
    
    # Apply scaling and offset for each channel
    radiance = np.empty(emissive.shape, dtype=np.float32)
    
    for i in range(emissive.shape[0]):
        channel_values = emissive[i, :, :]
        channel_radiance = ((channel_values.astype(np.float32) - attrs['radiance_offsets'][i]) /
                            attrs['radiance_scales'][i])
        # Handle special values
        channel_radiance[channel_values.astype(np.float32) == attrs['_FillValue']] = np.nan
        radiance[i, :, :] = channel_radiance

    return radiance

# Get the list of files in both directories
filesCC = sorted(os.listdir("CloudSatCALIPSO"))
filesMOD = sorted(os.listdir("MODIS_raw"))

y = np.empty((0,))
x = np.empty((0,22))

size = len(filesCC)
# Iterate over the files in CloudSat CALIPSO
for i in range(size):
 
    filenameCC = os.path.join("CloudSatCALIPSO",filesCC.pop(0))
    print(filenameCC)

    # Open HDF
    hdf_file = HDF(filenameCC, HC.READ)

    # Open Vdata interface
    vdata = hdf_file.vstart()

    field_ref = vdata.attach('Latitude')
    latitude_data_cc = field_ref[:]
    field_ref.detach()

    field_ref = vdata.attach('Longitude')
    longitude_data_cc = field_ref[:]
    field_ref.detach()

    field_ref = vdata.attach('Cloudlayer')
    cloud_layer = field_ref[:]
    field_ref.detach()

    # Close Vdata interface
    vdata.end()

    # Close HDF
    hdf_file.close()
            
    longitude_data_cc = np.ravel(longitude_data_cc)
    latitude_data_cc = np.ravel(latitude_data_cc)
    tree = build_tree(latitude_data_cc, longitude_data_cc)

    for j in range(20):

        # Check if there are files left in directory2
        if filesMOD:
            fnMOD = filesMOD.pop(0)
            # Take one file from directory2
            filenameMOD = os.path.join("MODIS_raw",fnMOD)
           # Open HDF
            hdf_file = SD(filenameMOD, SDC.READ)

            # List SDS datasets
            sds_datasets = hdf_file.datasets()

            latitude_data_mod = hdf_file.select('Latitude')[:]
            longitude_data_mod = hdf_file.select('Longitude')[:]

            radiance_1km = calcRadiance(hdf_file.select('EV_1KM_Emissive'))
            refradiance_1km = calcRadiance(hdf_file.select('EV_1KM_RefSB'))
            refradiance_250 = calcRadiance(hdf_file.select('EV_250_Aggr1km_RefSB'))
            refradiance_500 = calcRadiance(hdf_file.select('EV_500_Aggr1km_RefSB'))

            # Close the HDF file
            hdf_file.end()

            # Join radiances into feature space
            features = np.concatenate((radiance_1km, refradiance_1km, refradiance_250, refradiance_500), axis=0)

            # Grab middle lat/lons
            longitude_data_mod = np.ravel(longitude_data_mod[:,1])
            latitude_data_mod = np.ravel(latitude_data_mod[:,1])

            # Colocate CloudSat CALIPSO
            co_located_indices = co_locate(tree, latitude_data_mod, longitude_data_mod, 0.025)
            cloud_layer_colocated = np.array([cloud_layer[idx[0]] for idx in co_located_indices])
            cloud_layer_colocated = np.reshape(cloud_layer_colocated, (-1, 1))

            # Grab middle features
            features = features[:,2::5,5]

            # Colocate MODIS
            features_colocated = [features[:,idx[1]] for idx in co_located_indices]

            # If nothing is colocated, reset search on the next cloudsat calipso file
            if len(features_colocated) == 0:
                filesMOD.insert(0, fnMOD)
                break

            # Join array
            concatenated_array = np.concatenate((cloud_layer_colocated, features_colocated), axis=1)

            # Pato CSV file
            existing_csv_file = "data.csv"

            # Save NumPy array to temp CSV file
            np.savetxt("temp_array.csv", concatenated_array, delimiter=",")

            # Append temp CSV file to existing CSV file
            with open(existing_csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                with open("temp_array.csv", 'r') as temp_file:
                    reader = csv.reader(temp_file)
                    for row in reader:
                        writer.writerow(row)

            # Remove temp CSV file
            import os
            os.remove("temp_array.csv")


