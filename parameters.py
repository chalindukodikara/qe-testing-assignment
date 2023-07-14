################################################################
#################### Original Values ###########################
# WINDOW_SIZES = ['60', '50', '40', '30', '20', '10', '5', '3', '1']
# DEVICES = ["chest_band", 'earable', 'headband', 'wristband', 'earable_wristband', 'earable_chestband', 'wristband_chestband', 'wristband_chestband_earable']
# LABELLING_TYPES = ['person_centered_median', 'all_user_median', 'S1_vs_S3', 'S1_S2_vs_S3', 'S1_vs_S2_vs_S3']
# ML_MODELS = ['RandomForest', 'XGBoost', 'NeuralNetwork']
# MODEL_TYPES = ['population', 'hybrid']
CV_TYPES = ["leave_one_user_out", "leave_three_user_out"]
HYBRID_SPLIT_TYPES = ['random_split', 'session_split']

################################################################
########################## Variables############################
CV_TYPE = CV_TYPES[1]
HYBRID_SPLIT_TYPE = HYBRID_SPLIT_TYPES[1]
PCA = True # True or False
STANDARDIZE = True
DEVICES = ["chest_band", 'earable', 'headband', 'wristband', 'earable_wristband', 'earable_chestband', 'wristband_chestband', 'wristband_chestband_earable']
WINDOW_SIZES = ['60', '50', '40', '30', '20', '10', '5', '3', '1']
LABELLING_TYPES = ['person_centered_median', 'all_user_median', 'S1_vs_S3']
ML_MODELS = ['RandomForest', 'XGBoost', 'NeuralNetwork']
MODEL_TYPES = ['population']

FEATURE_CATEGORYWISE_ON = False # True or False, this will turn on feature categorywise code run

# Print details without running models
PRINT_NUM_OF_COLUMNS = False
SAVE_TEST_SET_WITH_DETAILS = False