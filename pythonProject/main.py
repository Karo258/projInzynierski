from zipfile import ZipFile
import radiomics.generalinfo
from radiomics import featureextractor
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
import numpy as np
import csv
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


# A function for creating indexes needed for maintaining files
def create_string_table(range_end):
    table = []
    for i in range(1, range_end):
        if i < 10:
            table.append('00' + str(i))
        elif i < 100:
            table.append('0' + str(i))
        else:
            table.append(str(i))
    return table


# A function for converting indexes to strings needed for comparison with string values in the string table
def convert_to_string(value):
    if value < 10:
        return '00' + str(value)
    elif value < 100:
        return '0' + str(value)
    else:
        return str(value)


# Unzipping the folder containing images
with ZipFile('C:/Users/Karolina/Desktop/MICCAI_BraTS2020_TrainingData.zip') as zipObj:
    zipObj.extractall()

# Table with numbers converted to string that are used while creating paths
numbers = create_string_table(370)

# Creating a numpy array of data
X = np.empty((368, 1), tuple)
for number in numbers:
    if number != '099':
        path_to_image = 'MICCAI_BraTS2020_TrainingData/BraTS20_Training_' + number + '/BraTS20_Training_' + number + '_flair.nii.gz'
        path_to_mask = 'MICCAI_BraTS2020_TrainingData/BraTS20_Training_' + number + '/BraTS20_Training_' + number + '_seg.nii.gz'
        image_mask_tuple = radiomics.featureextractor.RadiomicsFeatureExtractor.loadImage(path_to_image, path_to_mask)
        X.fill(image_mask_tuple)

# Creating a numpy array of "labels" -> HGG/LGG for stratifying data
y = np.hstack(([0] * 259, [1] * 76, [0] * 33))

name_mapping = pd.read_csv("MICCAI_BraTS2020_TrainingData/name_mapping.csv")
name_mapping_feature = ['Grade']
grades = name_mapping.loc[:, name_mapping_feature].values

# Creating an extractor for getting radiomic features
extractor = featureextractor.RadiomicsFeatureExtractor()

# Variables needed used for control of some functions in loops below
no_train_indexes = 0
iteration_train_index = 0
no_test_indexes = 0
iteration_test_index = 0

# Dividing data into two subsets: training and testing data, using Stratified KFold Cross-Validator
stratified_k_fold = StratifiedKFold(n_splits=3, shuffle=True)

# The loop for extracting features and writing them into .csv files
for train_index, test_index in stratified_k_fold.split(X, y):
    iteration_train_index += 1
    for i in train_index:
        no_train_indexes += 1
        if i < 98:
            number = convert_to_string(i + 1)
        else:
            number = convert_to_string(i + 2)
        path_to_image = 'MICCAI_BraTS2020_TrainingData/BraTS20_Training_' + number + '/BraTS20_Training_' + number + '_flair.nii.gz'
        path_to_mask = 'MICCAI_BraTS2020_TrainingData/BraTS20_Training_' + number + '/BraTS20_Training_' + number + '_seg.nii.gz'
        image_and_mask = radiomics.featureextractor.RadiomicsFeatureExtractor.loadImage(path_to_image, path_to_mask)
        image = image_and_mask[0]
        mask = image_and_mask[1]
        # Feature extraction
        extracted_features = extractor.execute(image, mask)
        # Writing extracted features into corresponding .csv files
        if no_train_indexes == 1:
            with open("extracted_features" + str(iteration_train_index) + ".csv", "w") as out_file:
                csv_writer = csv.writer(out_file)
                csv_writer.writerow(extracted_features)
                csv_writer.writerow(extracted_features.values())
            with open("glioma_grades" + str(iteration_train_index) + ".csv", "w") as out_file:
                csv_writer = csv.writer(out_file)
                csv_writer.writerow(grades[i])
        else:
            with open("extracted_features" + str(iteration_train_index) + ".csv", "a") as out_file:
                csv_writer = csv.writer(out_file)
                csv_writer.writerow(extracted_features.values())
            with open("glioma_grades" + str(iteration_train_index) + ".csv", "a") as out_file:
                csv_writer = csv.writer(out_file)
                csv_writer.writerow(grades[i])
    no_train_indexes = 0
    iteration_test_index += 1
    for i in test_index:
        no_test_indexes += 1
        if i < 98:
            number = convert_to_string(i + 1)
        else:
            number = convert_to_string(i + 2)
        path_to_image = 'MICCAI_BraTS2020_TrainingData/BraTS20_Training_' + number + '/BraTS20_Training_' + number + '_flair.nii.gz'
        path_to_mask = 'MICCAI_BraTS2020_TrainingData/BraTS20_Training_' + number + '/BraTS20_Training_' + number + '_seg.nii.gz'
        image_and_mask = radiomics.featureextractor.RadiomicsFeatureExtractor.loadImage(path_to_image, path_to_mask)
        image = image_and_mask[0]
        mask = image_and_mask[1]
        # Feature extraction
        extracted_features = extractor.execute(image, mask)
        if no_test_indexes == 1:
            with open("extracted_features_test" + str(iteration_test_index) + ".csv", "w") as out_file:
                csv_writer = csv.writer(out_file)
                csv_writer.writerow(extracted_features)
                csv_writer.writerow(extracted_features.values())
            with open("glioma_grades_test" + str(iteration_test_index) + ".csv", "w") as out_file:
                csv_writer = csv.writer(out_file)
                csv_writer.writerow(grades[i])
        else:
            with open("extracted_features_test" + str(iteration_test_index) + ".csv", "a") as out_file:
                csv_writer = csv.writer(out_file)
                csv_writer.writerow(extracted_features.values())
            with open("glioma_grades_test" + str(iteration_test_index) + ".csv", "a") as out_file:
                csv_writer = csv.writer(out_file)
                csv_writer.writerow(grades[i])
    no_test_indexes = 0

# The loop for reading .csv files and applying the PCA variance function
for i in range(1, iteration_train_index + 1):
    # df = pd.read_csv("extracted_features1.csv")
    data_file_train = pd.read_csv("extracted_features" + str(i) + ".csv")
    data_file_test = pd.read_csv("extracted_features_test" + str(i) + ".csv")
    data_file_grades_train = pd.read_csv("glioma_grades_train" + str(i) + ".csv")
    data_file_grades_test = pd.read_csv("glioma_grades_test" + str(i) + ".csv")

    # Features that are headers of float values in .csv files used for PCA
    features = ['diagnostics_Image-original_Mean', 'diagnostics_Image-original_Minimum', 'diagnostics_Image'
                                                                                         '-original_Maximum',
                'diagnostics_Mask-original_VoxelNum', 'diagnostics_Mask-original_VolumeNum',
                'original_shape_Elongation',
                'original_shape_Flatness', 'original_shape_LeastAxisLength', 'original_shape_MajorAxisLength',
                'original_shape_Maximum2DDiameterColumn', 'original_shape_Maximum2DDiameterRow',
                'original_shape_Maximum2DDiameterSlice', 'original_shape_Maximum3DDiameter',
                'original_shape_MeshVolume',
                'original_shape_MinorAxisLength', 'original_shape_Sphericity', 'original_shape_SurfaceArea',
                'original_shape_SurfaceVolumeRatio', 'original_shape_VoxelVolume', 'original_firstorder_10Percentile',
                'original_firstorder_90Percentile', 'original_firstorder_Energy', 'original_firstorder_Entropy',
                'original_firstorder_InterquartileRange', 'original_firstorder_Kurtosis', 'original_firstorder_Maximum',
                'original_firstorder_MeanAbsoluteDeviation', 'original_firstorder_Mean', 'original_firstorder_Median',
                'original_firstorder_Minimum', 'original_firstorder_Range',
                'original_firstorder_RobustMeanAbsoluteDeviation', 'original_firstorder_RootMeanSquared',
                'original_firstorder_Skewness', 'original_firstorder_TotalEnergy', 'original_firstorder_Uniformity',
                'original_firstorder_Variance', 'original_glcm_Autocorrelation', 'original_glcm_ClusterProminence',
                'original_glcm_ClusterShade', 'original_glcm_ClusterTendency', 'original_glcm_Contrast',
                'original_glcm_Correlation', 'original_glcm_DifferenceAverage', 'original_glcm_DifferenceEntropy',
                'original_glcm_DifferenceVariance', 'original_glcm_Id', 'original_glcm_Idm', 'original_glcm_Idmn',
                'original_glcm_Idn', 'original_glcm_Imc1', 'original_glcm_Imc2', 'original_glcm_InverseVariance',
                'original_glcm_JointAverage', 'original_glcm_JointEnergy', 'original_glcm_JointEntropy',
                'original_glcm_MCC', 'original_glcm_MaximumProbability', 'original_glcm_SumAverage',
                'original_glcm_SumEntropy', 'original_glcm_SumSquares', 'original_gldm_DependenceEntropy',
                'original_gldm_DependenceNonUniformity', 'original_gldm_DependenceNonUniformityNormalized',
                'original_gldm_DependenceVariance', 'original_gldm_GrayLevelNonUniformity',
                'original_gldm_GrayLevelVariance', 'original_gldm_HighGrayLevelEmphasis',
                'original_gldm_LargeDependenceEmphasis', 'original_gldm_LargeDependenceHighGrayLevelEmphasis',
                'original_gldm_LargeDependenceLowGrayLevelEmphasis', 'original_gldm_LowGrayLevelEmphasis',
                'original_gldm_SmallDependenceEmphasis', 'original_gldm_SmallDependenceHighGrayLevelEmphasis',
                'original_gldm_SmallDependenceLowGrayLevelEmphasis', 'original_glrlm_GrayLevelNonUniformity',
                'original_glrlm_GrayLevelNonUniformityNormalized', 'original_glrlm_GrayLevelVariance',
                'original_glrlm_HighGrayLevelRunEmphasis', 'original_glrlm_LongRunEmphasis',
                'original_glrlm_LongRunHighGrayLevelEmphasis', 'original_glrlm_LongRunLowGrayLevelEmphasis',
                'original_glrlm_LowGrayLevelRunEmphasis', 'original_glrlm_RunEntropy',
                'original_glrlm_RunLengthNonUniformity', 'original_glrlm_RunLengthNonUniformityNormalized',
                'original_glrlm_RunPercentage', 'original_glrlm_RunVariance', 'original_glrlm_ShortRunEmphasis',
                'original_glrlm_ShortRunHighGrayLevelEmphasis', 'original_glrlm_ShortRunLowGrayLevelEmphasis',
                'original_glszm_GrayLevelNonUniformity', 'original_glszm_GrayLevelNonUniformityNormalized',
                'original_glszm_GrayLevelVariance', 'original_glszm_HighGrayLevelZoneEmphasis',
                'original_glszm_LargeAreaEmphasis', 'original_glszm_LargeAreaHighGrayLevelEmphasis',
                'original_glszm_LargeAreaLowGrayLevelEmphasis', 'original_glszm_LowGrayLevelZoneEmphasis',
                'original_glszm_SizeZoneNonUniformity', 'original_glszm_SizeZoneNonUniformityNormalized',
                'original_glszm_SmallAreaEmphasis', 'original_glszm_SmallAreaHighGrayLevelEmphasis',
                'original_glszm_SmallAreaLowGrayLevelEmphasis', 'original_glszm_ZoneEntropy',
                'original_glszm_ZonePercentage', 'original_glszm_ZoneVariance', 'original_ngtdm_Busyness',
                'original_ngtdm_Coarseness', 'original_ngtdm_Complexity', 'original_ngtdm_Contrast',
                'original_ngtdm_Strength']
X_train = data_file_train.loc[:, features].values
X_test = data_file_test.loc[:, features].values
Y_train = data_file_grades_train.values
Y_test = data_file_grades_test.values
X_train = StandardScaler().fit_transform(X_train)
X_test = StandardScaler().fit_transform(X_test)
# Y_train = StandardScaler().fit_transform(Y_train)
# Y_test = StandardScaler().fit_transform(Y_test)
pca = PCA(n_components=29)  # do wyjaśnienia 99% wariancji potrzebnych jest 29 cech
X_train = pca.fit_transform(X_train)
explained_variance = pca.explained_variance_ratio_
classifier = RandomForestClassifier(max_depth=2)
classifier.fit(X_train, Y_train)

Y_predict = classifier.predict(X_test)
print(Y_predict )
