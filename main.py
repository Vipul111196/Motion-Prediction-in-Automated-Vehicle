# Importing Required Modules
import os
import sys
import xlsxwriter
from sklearn import preprocessing
import numpy as np
import pandas as pd
from src.data_processing.data_processing_classes.readDataset import dataGrabber
from src.data_processing.data_processing_classes.preProcessing import preProcess
from src.data_processing.data_processing_classes.dataPreparation import dataPrepare
from src.prediction_models.neural_networks.fcn_keras_model import FCN_keras_model
from src.evaluation.evaluation_classes.data_driven_pred_evaluator import dataDrivenEvaluation
from src.evaluation.evaluation_classes.evaluation_matrix import evaluationMatrix
import yaml

print('\n')
print('************************************  Program Started!!  ************************************')
print('\n')
print("Libraries Succesfully imported!!")

# Load the parameters
params = yaml.safe_load(open('params.yaml'))

# Data Preprocessing
if params['training_data_preprocessed_already'] == False:
    dataset_path = params['dataset_path']
    print(dataset_path)
    print('Dataset path loaded successfully!!')

    # Reading the dataset
    recording_id_sel = params['recording_id_sel']
    print("Recording ID Selected")

    # Initialize data Grabber Object
    data_obj = dataGrabber(dataset_path)

    data_obj.recording_id = recording_id_sel
    data_obj.read_csv_with_recordingID()

    track_data_raw = data_obj.get_tracks_data()
    track_meta_data_raw = data_obj.get_tracksMeta_data()  

    print("Main Data Grabbed Successfully from entire dataset!!")

    # Initialize Data Preparation Object
    pre_process_obj = preProcess()
    pre_process_obj.tracks_data = track_data_raw
    pre_process_obj.tracks_meta_data = track_meta_data_raw
    pre_process_obj.recording_ids = data_obj.recording_id
    pre_process_obj.data_len = len(track_data_raw)

    print("Data Preprocessing Object Created!!")
    print('Data Preprocessing Started!!')
    print('\n')

    # Preprocess the data
    # Define the number of frames to be skipped + 1 => here 4 frames are skipped so 4+1 = 5

    pre_process_obj.frames_skipped = 5
    track_data_downsampled, tracks_meta_data = pre_process_obj.get_down_sampled_data()
    pre_process_obj.tracks_data = track_data_downsampled
    print('Frames skipped to downsample the data')

    # Label Encoding
    pre_process_obj.label_encoding()
    pre_process_obj.print_label_encoder_classes()
    print('Frames skipped to downsample the data --> Label Encoding Done!!')

    # Normalize the data
    # Gets the tracks data normalized
    tracks_data_norm, min_max_scalar_list = pre_process_obj.normalize_data()
    print('Frames skipped to downsample the data --> Label Encoding Done --> Data Normalized!!')

    # Resetting dropped frames index
    tracks_data_norm = tracks_data_norm.reset_index(drop=True)

    data_prepare_obj = dataPrepare()
    data_prepare_obj.tracks_data_norm = tracks_data_norm
    data_prepare_obj.tracksMeta_data = tracks_meta_data
    data_prepare_obj.data_len = len(tracks_data_norm)

    # Splitting the data
    # Number for track id to be used
    data_prepare_obj.track_id_range = 100

    data_prepare_obj.data_input = "normalized_data"
    xTrain_data, xTest_data, yTrain_data, yTest_data = data_prepare_obj.get_test_train_split()
    print('Frames skipped to downsample the data --> Label Encoding Done --> Data Normalized --> Data Split!!')

    # Saving the preprocessed data
    path_to_save_data = params['path_to_save_and_load_data_pickle']
    # Save the xTrain, xTest, yTrain, xTest in pickle format
    data_prepare_obj.save_test_train_data_pickle(path_to_save_data)
    print(f"Data Saved Successfully to path {path_to_save_data}!!")
    print(f"Length of x training data {len(data_prepare_obj.xTrain_data)}")
    print(f"Length of y training data {len(data_prepare_obj.yTrain_data)}")
    print(f"Length of x testing data {len(data_prepare_obj.xTest_data)}")
    print(f"Length of y testing data {len(data_prepare_obj.yTest_data)}")

if params['training_data_preprocessed_already'] == True:
    data_prepare_obj = dataPrepare()
    path_to_save_data = params['path_to_save_and_load_data_pickle']
    print('Data already processed --> Data Preprocessing Skipped!!')
    # Save the xTrain, xTest, yTrain, xTest in pickle format
    data_prepare_obj.load_test_train_data_pickle(path_to_save_data)
    print(f"Data Loaded Successfully from path {path_to_save_data}!!")
    print(f"Length of x training data {len(data_prepare_obj.xTrain_data)}")
    print(f"Length of y training data {len(data_prepare_obj.yTrain_data)}")
    print(f"Length of x testing data {len(data_prepare_obj.xTest_data)}")
    print(f"Length of y testing data {len(data_prepare_obj.yTest_data)}")

print('\n')
print('************************************  Data preprocessing stage completed!!  ************************************')
print('\n')

# Initialize the FCN Keras Model training
print('Model Training Started!!')
model_path = params['model_path']
xTrain_data = data_prepare_obj.xTrain_data
xTest_data = data_prepare_obj.xTest_data
yTrain_data = data_prepare_obj.yTrain_data
yTest_data = data_prepare_obj.yTest_data
FCN_model =  FCN_keras_model(xTrain_data, xTest_data, yTrain_data, yTest_data, model_path)
print('Model Training Completed!!')
print(f"Model saved to path {model_path}!!")

print('\n')
print('************************************  Model Training stage completed!!  ************************************')
print('\n')

# Prediction Testing
print('Prediction Testing Started!!')
if params['testing_data_preprocessed_already']==False:
    dataset_path = params['dataset_path']
    print(dataset_path)
    print('Testing Dataset path loaded successfully!!')

    # Initialize data Grabber Object
    test_data_obj = dataGrabber(dataset_path)

    recording_id_sel = params['recording_id_sel_test']
    test_data_obj.recording_id = recording_id_sel
    test_data_obj.read_csv_with_recordingID()

    test_track_data = test_data_obj.get_tracks_data()
    test_track_meta_data = test_data_obj.get_tracksMeta_data() 
    print("Testing Data Grabbed Successfully!!")

    test_pre_process_obj = preProcess()
    test_pre_process_obj.tracks_data = test_track_data
    test_pre_process_obj.tracks_meta_data = test_track_meta_data
    test_pre_process_obj.recording_ids = test_data_obj.recording_id
    test_pre_process_obj.data_len = len(test_track_data)

    print("Data Preprocessing Object Created!!")
    print('Data Preprocessing Started!!')
    print('\n')
    
    test_pre_process_obj.frames_skipped = 5
    test_track_data_downsampled, test_tracks_meta_data = test_pre_process_obj.get_down_sampled_data()
    test_pre_process_obj.tracks_data = test_track_data_downsampled
    print('Frames skipped to downsample the data')

    # Gets the tracks data normalized
    test_tracks_data_norm, min_max_scalar_list = test_pre_process_obj.normalize_data()
    print('Frames skipped to downsample the data --> Data Normalized!!')

    # Resetting dropped frames index
    test_tracks_data_norm = test_tracks_data_norm.reset_index(drop=True)
    
    # Saving Normalized Data
    test_data_prepare_obj = dataPrepare()
    test_data_prepare_obj.tracks_data_norm = test_tracks_data_norm
    test_data_prepare_obj.tracksMeta_data = test_tracks_meta_data
    test_data_prepare_obj.data_len = len(test_tracks_data_norm)
    print('Data Normalized Successfully!!')

    # Number for track id to be used
    test_data_prepare_obj.track_id_range = 10  
    
    # Gets the tracks data normalized and its ID
    test_data_prepare_obj.data_input = "normalized_data"
    t_norm_Ids, t_in_norm, t_out_norm = test_data_prepare_obj.data_stacking()

    print('Test Data Stacked Successfully!!')
    # Predict the output
    n_input = np.shape(t_in_norm)[1] * np.shape(t_in_norm)[2]
    t_in_norm_reshaped = np.reshape(t_in_norm, (np.shape(t_in_norm)[0], n_input))
    print('Test Data Stacked Successfully!!')

    # Save the preprocessed data
    path_to_save_test_data = params['path_to_save_and_load_test_data_pickle']
    # Save the xTrain, xTest, yTrain, xTest in pickle format
    test_data_prepare_obj.save_test_train_data_pickle(path_to_save_test_data)
    print(f"Test Data Saved Successfully to path {path_to_save_test_data}!!")

if params['testing_data_preprocessed_already']==True:
    test_data_prepare_obj = dataPrepare()
    path_to_save_test_data = params['path_to_save_and_load_test_data_pickle']
    print('Test Data already processed --> Data Preprocessing Skipped!!')
    # Save the xTrain, xTest, yTrain, xTest in pickle format
    test_data_prepare_obj.load_test_train_data_pickle(path_to_save_test_data)
    print(f"Test Data Loaded Successfully from path {path_to_save_test_data}!!")

print('\n')
print('************************************  Testing data preparation stage completed!!  ************************************')
print('\n')

# Collect Ground Truth Data
data_eval_obj = dataDrivenEvaluation()

# Resetting dropped frames index
test_track_data_downsampled = test_track_data_downsampled.reset_index(drop=True)
ground_truth_prepare_obj = dataPrepare()
ground_truth_prepare_obj.data_input = "raw_data"
ground_truth_prepare_obj.track_id_range = 10
ground_truth_prepare_obj.tracksMeta_data = test_tracks_meta_data
ground_truth_prepare_obj.tracks_data_norm = test_tracks_data_norm
ground_truth_prepare_obj.tracks_data = test_track_data_downsampled
ground_truth_prepare_obj.data_len = len(test_track_data_downsampled) 
#ground_truth_prepare_obj.num_predict = 15
t_raw_Ids, t_in_raw, t_out_raw = ground_truth_prepare_obj.data_stacking()

# Get the ground truth data
data_eval_obj.t_raw_Ids = t_raw_Ids
data_eval_obj.t_in_raw = t_in_raw
data_eval_obj.t_out_raw = t_out_raw

xCenter_gt, yCenter_gt, heading_gt = data_eval_obj.get_ground_truth()

# Predict the output
yhat = FCN_model.predict(t_in_norm_reshaped, verbose=0)
    
# Save Predicted Data into the Evaluator
data_eval_obj.y_hat = yhat

# Set Paramters
data_eval_obj.min_max_scalar_list = min_max_scalar_list

# Get Prediction
xCenter_prediction, yCenter_prediction, heading_prediction = data_eval_obj.get_prediction()

# Save the Predicted Data
work_book_filename = params['work_book_filename']
if os.path.exists(work_book_filename):
    os.remove(work_book_filename)
print(f"File {work_book_filename} removed successfully!!")

data_eval_obj.wb_filename = work_book_filename
data_eval_obj.write_to_workbook()

# Final Evaluation
eval_obj = evaluationMatrix(work_book_filename, data_eval_obj.n_predict) 
# Print the Evaluation Matrix
ade_value, fde_value, ahe_val = eval_obj.get_fde_ade_ahe()

# Save the Results
results_path = params['results_path']
with open(results_path, 'w') as file:
    file.write(f"ADE Value: {ade_value}\n")
    file.write(f"FDE Value: {fde_value}\n")
    file.write(f"AHE Value: {ahe_val}\n")
print(f"Results saved to path {results_path}!!")

print('Prediction Testing Completed!!')

print('Program Completed Successfully!!')
print('Exiting the Program!!')
print('\n')
print('************************************  Thank You!!  ************************************')
