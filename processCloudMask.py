from pyhdf.SD import SD, SDC
import os
import numpy as np
from pyhdf.HDF import *
from pyhdf.VS import *
from scipy.spatial import cKDTree
import csv

def build_tree(lat1, lon1):
    """
    Build a KDTree from latitude and longitude arrays.
    
    Parameters:
        lat1 (ndarray): Latitude array.
        lon1 (ndarray): Longitude array.
    
    Returns:
        tree: KDTree object.
    """
    return cKDTree(np.column_stack((lat1, lon1)))

def co_locate(tree, lat2, lon2, threshold):
    co_located_indices = []
    
    for i, (lat, lon) in enumerate(zip(lat2, lon2)):
        dist, idx = tree.query([lat, lon])
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

    # Open the HDF file in read mode
    hdf_file = HDF(filenameCC, HC.READ)

    # Open the Vdata interface
    vdata = hdf_file.vstart()

    # Initialize latitude and longitude data
    latitude_data = None
    longitude_data = None

    # Get the list of Vdata fields
    vdata_info = vdata.vdatainfo()

    # Iterate through each Vdata field
    for index, info in enumerate(vdata_info):
        try:
            field_name = info[0]  # Name of the Vdata field
            
            # Check if the Vdata field contains latitude or longitude data
            if field_name == 'Latitude':
                field_ref = vdata.attach(field_name)  # Reference to the Vdata field
                latitude_data_cc = field_ref[:]
                # Detach from the current Vdata field
                field_ref.detach()
            elif field_name == 'Longitude':
                field_ref = vdata.attach(field_name)  # Reference to the Vdata field
                longitude_data_cc = field_ref[:]
                # Detach from the current Vdata field
                field_ref.detach()
            elif field_name == 'Cloudlayer':
                field_ref = vdata.attach(field_name)  # Reference to the Vdata field
                cloud_layer = field_ref[:]
                cloud_layer = np.ravel(cloud_layer)
                # Detach from the current Vdata field
                field_ref.detach()
            
        except HDF4Error as e:
            print(f"Error attaching Vdata field at index {index}: {e}")

    # Close the Vdata interface
    vdata.end()

    # Close the HDF file
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
            # Open the HDF file in read mode
            hdf_file = SD(filenameMOD, SDC.READ)

            # List SDS datasets in the file
            sds_datasets = hdf_file.datasets()

            # Iterate through SDS datasets
            for dataset_name in sds_datasets:
                # print("SD dataset name:", dataset_name)
                # Check if the dataset contains geolocation information
                if 'Latitude' in dataset_name:
                    latitude_data_mod = hdf_file.select(dataset_name)[:]
                elif 'Longitude' in dataset_name:
                    longitude_data_mod = hdf_file.select(dataset_name)[:]
                elif 'Cloud_Mask' == dataset_name:
                    data = hdf_file.select(dataset_name)
                    dataarray = data[:]
                    dataarray = dataarray[0,:,5]

                    binary_data = [format(byte, '08b') for byte in dataarray]
                    extracted_bits = [byte[-7:-5] for byte in binary_data]
                    cloud_mask_mod = [1 if bits in ['00', '01'] else 0 for bits in extracted_bits]

            # Close the HDF file
            hdf_file.end()
            longitude_data_mod = np.ravel(longitude_data_mod[:,1])
            latitude_data_mod = np.ravel(latitude_data_mod[:,1])
            co_located_indices = co_locate(tree, latitude_data_mod, longitude_data_mod, 0.025)
            cloud_layer_colocated = np.array([cloud_layer[idx[0]] for idx in co_located_indices])
            cloud_layer_colocated = np.reshape(cloud_layer_colocated, (-1, 1))
            cloud_mask_mod_colocated = [cloud_mask_mod[idx[1]] for idx in co_located_indices]
            cloud_mask_mod_colocated = np.reshape(cloud_mask_mod_colocated, (-1, 1))

            if len(cloud_mask_mod_colocated) == 0:
                filesMOD.insert(0, fnMOD)
                break

            concatenated_array = np.concatenate((cloud_layer_colocated, cloud_mask_mod_colocated), axis=1)

            # Path to the existing CSV file
            existing_csv_file = "modMaskData.csv"

            # Save the NumPy array to a temporary CSV file
            np.savetxt("temp_array.csv", concatenated_array, delimiter=",")

            # Append the temporary CSV file to the existing CSV file
            with open(existing_csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                with open("temp_array.csv", 'r') as temp_file:
                    reader = csv.reader(temp_file)
                    for row in reader:
                        writer.writerow(row)

            # Optionally, you can remove the temporary CSV file
            import os
            os.remove("temp_array.csv")


