# Quantum Machine Learning for Cloud Detection

## Folder Names

1. Preproessing

Jupyter Lab files include preprocessing of just one MODIS/C-C file
The python files will preprocess all of the files within MODIS_raw/MODIS_cloudMask (depending on the file) & CloudSatCALIPSO. Results are saved as either data.csv or modMaskData.csv

2. Machine Learning

Jupyter Lab files for each Machine Learning model used in the poster

3. Visualization

Jupyter Lab files for each visualization made for the poster
Some other visualizations on the poster were in the preprocessing file
Lastly some images were gathered directly from NASA's database. Their resources were included on the poster
  
4. MODIS_raw

Includes some examples of MAC021S0 files
  
5. MODIS_cloudMask

Includes some examples of MAC035S0 files
   
6. CloudSatCALIPSO

Includes some examples of 2B-CLDCLASS-LIDAR
  
7. Saved_ML_Models

Where ML models are saved for visualization later

## Abstract

Most satellite imagery requires an indicator of when a cloud covers its observation, named a “cloud mask”. NASA’s Aqua and Terra satellites use the MODIS (Moderate Resolution Imaging Spectroradiometer) sensor to develop their cloud masks. NASA uses a series of physics equations to detect and flag clouds. Unfortunately, these physics algorithms have weaknesses. It does badly at nighttime, has trouble when snow covers the ground, and is lost when the sun reflects into the satellite’s sensor (also known as sun-glint).
Compared to more specialized sensors aboard the CloudSat and CALIPSO satellites, MODIS’ cloud detection is average. Using raw MODIS data, researchers of cloud detection have turned to machine learning for better results. The Random Forest (RF) and Artificial Neural Network (ANN) approaches are popular for outpacing physics algorithms. With the introduction of quantum computing to RPI, big data problems such as satellite imaging can be trained with a Quantum Support Vector Machine (QSVM). This poster dives into quantum machine-learning methods and compares them with traditional machine-learning models.
An SVM, RF, ANN, and QSVM were trained to predict clouds. The models used MODIS’ radiance bands as features and CC cloud mask as a target variable. The SVMs could not run with all ~140k data points & 22 features. The RF achieved 92.4% accuracy and the ANN got 88.9% accuracy. These accuracies are impressive against MODIS’ cloud mask which achieved 45% accuracy vs CC. When downsampling to 1,000 data points and 2 features, the SVM and QSVM were run. SVM 77.4%, ANN 77.3%, RF 74.8%, QSVM 74.8%. The SVM models are competitive when downsampled. With the introduction of a Quantum Computer to RPI, our QSVM model will be trainable on the large sample set. Next, we hope to train on the Quantum RF model.
