from zipfile import ZipFile
import radiomics.generalinfo
import six
from radiomics import featureextractor, firstorder
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
import numpy as np
import csv
import pandas as pd
from sklearn.preprocessing import StandardScaler


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

# Creating an extractor for getting radiomic features
extractor = featureextractor.RadiomicsFeatureExtractor()

# no_train_indexes = 0
# # Dividing data into two subsets: training and testing data, using Stratified KFold Cross-Validator
# stratified_k_fold = StratifiedKFold(n_splits=3, shuffle=True)
# iteration = 0
# for train_index, test_index in stratified_k_fold.split(X, y):
#     iteration += 1
#     keys, values, tmp = [], [], []
#     for i in train_index:
#         no_train_indexes += 1
#         if i < 98:
#             number = convert_to_string(i + 1)
#         else:
#             number = convert_to_string(i + 2)
#         path_to_image = 'MICCAI_BraTS2020_TrainingData/BraTS20_Training_' + number + '/BraTS20_Training_' + number + '_flair.nii.gz'
#         path_to_mask = 'MICCAI_BraTS2020_TrainingData/BraTS20_Training_' + number + '/BraTS20_Training_' + number + '_seg.nii.gz'
#         image_and_mask = radiomics.featureextractor.RadiomicsFeatureExtractor.loadImage(path_to_image, path_to_mask)
#         image = image_and_mask[0]
#         mask = image_and_mask[1]
#         extracted_features = extractor.execute(image, mask)
#         if no_train_indexes == 1:
#             with open("extracted_features" + str(iteration) +".csv", "w") as outfile:
#                 csvwriter = csv.writer(outfile)
#                 csvwriter.writerow(extracted_features)
#                 csvwriter.writerow(extracted_features.values())
#         else:
#             with open("extracted_features" + str(iteration) +".csv", "a") as outfile:
#                 csvwriter = csv.writer(outfile)
#                 csvwriter.writerow(extracted_features.values())
#     no_train_indexes = 0

df = pd.read_csv("extracted_features1.csv")

features = ['diagnostics_Image-original_Mean', 'diagnostics_Image-original_Minimum', 'diagnostics_Image-original_Maximum', 'diagnostics_Mask-original_VoxelNum', 'diagnostics_Mask-original_VolumeNum', 'original_shape_Elongation', 'original_shape_Flatness', 'original_shape_LeastAxisLength', 'original_shape_MajorAxisLength', 'original_shape_Maximum2DDiameterColumn', 'original_shape_Maximum2DDiameterRow', 'original_shape_Maximum2DDiameterSlice', 'original_shape_Maximum3DDiameter', 'original_shape_MeshVolume', 'original_shape_MinorAxisLength', 'original_shape_Sphericity', 'original_shape_SurfaceArea', 'original_shape_SurfaceVolumeRatio', 'original_shape_VoxelVolume', 'original_firstorder_10Percentile', 'original_firstorder_90Percentile', 'original_firstorder_Energy', 'original_firstorder_Entropy', 'original_firstorder_InterquartileRange', 'original_firstorder_Kurtosis', 'original_firstorder_Maximum', 'original_firstorder_MeanAbsoluteDeviation', 'original_firstorder_Mean', 'original_firstorder_Median', 'original_firstorder_Minimum', 'original_firstorder_Range', 'original_firstorder_RobustMeanAbsoluteDeviation', 'original_firstorder_RootMeanSquared', 'original_firstorder_Skewness', 'original_firstorder_TotalEnergy', 'original_firstorder_Uniformity', 'original_firstorder_Variance', 'original_glcm_Autocorrelation', 'original_glcm_ClusterProminence', 'original_glcm_ClusterShade', 'original_glcm_ClusterTendency', 'original_glcm_Contrast', 'original_glcm_Correlation', 'original_glcm_DifferenceAverage', 'original_glcm_DifferenceEntropy', 'original_glcm_DifferenceVariance', 'original_glcm_Id', 'original_glcm_Idm', 'original_glcm_Idmn', 'original_glcm_Idn', 'original_glcm_Imc1', 'original_glcm_Imc2', 'original_glcm_InverseVariance', 'original_glcm_JointAverage', 'original_glcm_JointEnergy', 'original_glcm_JointEntropy', 'original_glcm_MCC', 'original_glcm_MaximumProbability', 'original_glcm_SumAverage', 'original_glcm_SumEntropy', 'original_glcm_SumSquares', 'original_gldm_DependenceEntropy', 'original_gldm_DependenceNonUniformity', 'original_gldm_DependenceNonUniformityNormalized', 'original_gldm_DependenceVariance', 'original_gldm_GrayLevelNonUniformity', 'original_gldm_GrayLevelVariance', 'original_gldm_HighGrayLevelEmphasis', 'original_gldm_LargeDependenceEmphasis', 'original_gldm_LargeDependenceHighGrayLevelEmphasis', 'original_gldm_LargeDependenceLowGrayLevelEmphasis', 'original_gldm_LowGrayLevelEmphasis', 'original_gldm_SmallDependenceEmphasis', 'original_gldm_SmallDependenceHighGrayLevelEmphasis', 'original_gldm_SmallDependenceLowGrayLevelEmphasis', 'original_glrlm_GrayLevelNonUniformity', 'original_glrlm_GrayLevelNonUniformityNormalized', 'original_glrlm_GrayLevelVariance', 'original_glrlm_HighGrayLevelRunEmphasis', 'original_glrlm_LongRunEmphasis', 'original_glrlm_LongRunHighGrayLevelEmphasis', 'original_glrlm_LongRunLowGrayLevelEmphasis', 'original_glrlm_LowGrayLevelRunEmphasis', 'original_glrlm_RunEntropy', 'original_glrlm_RunLengthNonUniformity', 'original_glrlm_RunLengthNonUniformityNormalized', 'original_glrlm_RunPercentage', 'original_glrlm_RunVariance', 'original_glrlm_ShortRunEmphasis', 'original_glrlm_ShortRunHighGrayLevelEmphasis', 'original_glrlm_ShortRunLowGrayLevelEmphasis', 'original_glszm_GrayLevelNonUniformity', 'original_glszm_GrayLevelNonUniformityNormalized', 'original_glszm_GrayLevelVariance', 'original_glszm_HighGrayLevelZoneEmphasis', 'original_glszm_LargeAreaEmphasis', 'original_glszm_LargeAreaHighGrayLevelEmphasis', 'original_glszm_LargeAreaLowGrayLevelEmphasis', 'original_glszm_LowGrayLevelZoneEmphasis', 'original_glszm_SizeZoneNonUniformity', 'original_glszm_SizeZoneNonUniformityNormalized', 'original_glszm_SmallAreaEmphasis', 'original_glszm_SmallAreaHighGrayLevelEmphasis', 'original_glszm_SmallAreaLowGrayLevelEmphasis', 'original_glszm_ZoneEntropy', 'original_glszm_ZonePercentage', 'original_glszm_ZoneVariance', 'original_ngtdm_Busyness', 'original_ngtdm_Coarseness', 'original_ngtdm_Complexity', 'original_ngtdm_Contrast', 'original_ngtdm_Strength']
# features = values.head(0)

x = df.loc[:, features].values

# print('przed -> ', x)
x = StandardScaler().fit_transform(x)
pca = PCA(n_components=10)
principalComponents = pca.fit_transform(x)
print(principalComponents)
