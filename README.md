# Transfer Learning for Brain Tumor Segmentation

This is the repository containing the code to reproduce the experiments in ``Transfer Learning for Brain Tumor Segmentation`` (arXiv: ...).

## Requirements to run the code:
- BraTS '17/'18 Training Data and BraTS '18 Validation Data
-- See here for instructions how to obtain the data: http://braintumorsegmentation.org/
-- The BraTS '18 validation data is used to obtain test results. Predictions can of course be carried out on any other dataset.
- PyTorch
- batchgenerators by MIC@DKFZ (https://github.com/MIC-DKFZ/batchgenerators)
-- Follow the following guide in order to convert the BRATS data into the necessary format:
-- https://github.com/MIC-DKFZ/batchgenerators/tree/master/batchgenerators/examples/brats2017
-- The data should be stored in brats_data_preprocessed/Brats17TrainingData (and Brats18ValidationData)
-- Be aware that the new output format occupies considerably more disk space! E.g. 18 GB for the BraTS 17 training data!

## Files and folders inside this repository:
- brats_data_preprocessed: The preprocessed BraTS data stored in a separate subdirectory for each year and type (train/validation)
- models: The models saved by PyTorch
- segmentation_output: The output segmentations produced by the trained model in NIFTI format. These can be directly uploaded to the BraTS evaluation server.
- tensorboard_logs: Tensorboard logfiles that contain the dice scores/losses over time.
- Read-Logs.ipynb: Notebook to visualize the tensorboard logs
- brats_data_loader.py: Wrapper class for the BraTS dataloader used to train the model from the preprocessed files.
- jonas_net.py: Contains the AlbuNet3D architecture using a ResNet34 encoder.
- tb_log_reader.py: Wrapper class to read tensorboard logs.
- ternaus_unet_models.py: Reference file containing the original AlbuNet model.
- train_jonas_net_batch.py: Python script to train the model for a given configuration passed as arguments.
- train_test_function.py: Helper class to facilitate the training procedure for any deep learning model.

- run_experiments_x.sh: Shell script to launch train_jonas_net_batch.py for the configurations used in the paper.