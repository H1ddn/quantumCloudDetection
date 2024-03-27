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

# Get the list of files in both directories
filesCC = sorted(os.listdir("CloudSatCALIPSO"))
filesMOD = sorted(os.listdir("MODIS_cloudMask"))

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
            filenameMOD = os.path.join("MODIS_cloudMask",fnMOD)
            # Open HDF
            hdf_file = SD(filenameMOD, SDC.READ)


            latitude_data_mod = hdf_file.select('Latitude')[:]
            longitude_data_mod = hdf_file.select('Longitude')[:]
    
            # Grab 0th end byte
            dataarray = hdf_file.select('Cloud_Mask')[0,:,5]
            # Gather data as Bytes
            binary_data = [format(byte & 0xFF, '08b') for byte in dataarray]
            # Gather 2nd and 3rd bit from right
            extracted_bits = [byte[-3:-1] for byte in binary_data]
            # Cloud mask information, simplify to [1,0]
            cloud_mask_mod = [0 if bits in ['11'] else 1 for bits in extracted_bits]
                    
            # Close HDF
            hdf_file.end()

            longitude_data_mod = np.ravel(longitude_data_mod[:,1])
            latitude_data_mod = np.ravel(latitude_data_mod[:,1])
            co_located_indices = co_locate(tree, latitude_data_cc, longitude_data_cc, latitude_data_mod, longitude_data_mod, 0.025)
            cloud_layer_colocated = np.array([cloud_layer[idx[0]] for idx in co_located_indices])
            cloud_layer_colocated = np.reshape(cloud_layer_colocated, (-1, 1))
            cloud_mask_mod_colocated = [cloud_mask_mod[idx[1]] for idx in co_located_indices]
            cloud_mask_mod_colocated = np.reshape(cloud_mask_mod_colocated, (-1, 1))

            if len(cloud_mask_mod_colocated) == 0:
                filesMOD.insert(0, fnMOD)
                break

            concatenated_array = np.concatenate((cloud_layer_colocated, cloud_mask_mod_colocated), axis=1)

            # Path CSV file
            existing_csv_file = "modMaskData.csv"

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


