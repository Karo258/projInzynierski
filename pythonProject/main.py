from zipfile import ZipFile
import radiomics.generalinfo
import six
from radiomics import featureextractor, firstorder
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
import numpy as np


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

no_train_indexes = 0
# Dividing data into two subsets: training and testing data, using Stratified KFold Cross-Validator
stratified_k_fold = StratifiedKFold(n_splits=3, shuffle=True)
for train_index, test_index in stratified_k_fold.split(X, y):
    # numpyArray = np.array((129, 2), dtype=object)
    numpyArray2 = np.array((245, 1), dtype=object)
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
        extracted_features = extractor.execute(image, mask)
        # for key, value in six.iteritems(extracted_features):
        #     print(key, value)
        # result = extracted_features.items()
        # data = list(result)
        # numpyArray.fill(data)
        # numpyArray = np.array(data)
        # print(numpyArray)
        numpyArray2.fill(extracted_features)
    print('pierwsza')
    pca = PCA(n_components=2)
    X_new = pca.fit_transform(numpyArray2)
    print('druga')
    # print(X_new)
    no_train_indexes = 0
