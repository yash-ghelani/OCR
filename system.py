"""Dummy classification system.

Skeleton code for a assignment solution.

To make a working solution you will need to rewrite parts
of the code below. In particular, the functions
reduce_dimensions and classify_page currently have
dummy implementations that do not do anything useful.

version: v1.0
"""
import numpy as np
import utils.utils as utils
import scipy.linalg
from scipy import ndimage
import re

def multidivergence(class1, class2, features):
    """compute divergence between class1 and class2
    class1 - data matrix for class 1, each row is a sample
    class2 - data matrix for class 2
    features - the subset of features to use
    returns: d12 - a scalar divergence score
    """

    ndim = len(features);

    # compute mean vectors
    mu1 = np.mean(class1[:, features], axis=0)
    mu2 = np.mean(class2[:, features], axis=0)

    # compute distance between means
    dmu = mu1 - mu2

    # compute covariance and inverse covariance matrices
    cov1 = np.cov(class1[:, features], rowvar=0)
    cov2 = np.cov(class2[:, features], rowvar=0)
 
    icov1 = np.linalg.inv(cov1)
    icov2 = np.linalg.inv(cov2)

    # plug everything into the formula for multivariate gaussian divergence
    d12 = (0.5 * np.trace(np.dot(icov1, cov2) + np.dot(icov2, cov1) 
                          - 2 * np.eye(ndim)) + 0.5 * np.dot(np.dot(dmu, icov1 + icov2), dmu))

    return d12

def feature_selection(features):
    #multidivergence(features)
    return features[:, 0:10]

def reduce_dimensions(feature_vectors_full, model):
    """
    Params:
    feature_vectors_full - feature vectors stored as rows
       in a matrix
    model - a dictionary storing the outputs of the model
       training stage
    """
    # Getting all eigenvectors calculated in process_training_data for PCA
    v = model['eigenvectors']

    # Projecting data onto PCA
    pcatest_data = np.dot((feature_vectors_full - np.mean(feature_vectors_full, axis = 0)), v)

    # Returns best 10 pixels
    best_features = feature_selection(pcatest_data)

    return best_features


def get_bounding_box_size(images):
    """Compute bounding box size given list of images."""

    for image in images:
        image = ndimage.median_filter(image,3)

    height = max(image.shape[0] for image in images)
    width = max(image.shape[1] for image in images)
    return height, width



def images_to_feature_vectors(images, bbox_size=None):
    """Reformat characters into feature vectors.

    Takes a list of images stored as 2D-arrays and returns
    a matrix in which each row is a fixed length feature vector
    corresponding to the image.abs

    Params:
    images - a list of images stored as arrays
    bbox_size - an optional fixed bounding box size for each image
    """

    # If no bounding box size is supplied then compute a suitable
    # bounding box by examining sizes of the supplied images.
    if bbox_size is None:
        bbox_size = get_bounding_box_size(images)

    bbox_h, bbox_w = bbox_size
    nfeatures = bbox_h * bbox_w
    fvectors = np.empty((len(images), nfeatures))
    for i, image in enumerate(images):
        ####### Image filtering #####
        image = ndimage.median_filter(image,3)
        padded_image = np.ones(bbox_size) * 255
        h, w = image.shape
        h = min(h, bbox_h)
        w = min(w, bbox_w)
        padded_image[0:h, 0:w] = image[0:h, 0:w]
        fvectors[i, :] = padded_image.reshape(1, nfeatures)
    return fvectors


# The three functions below this point are called by train.py
# and evaluate.py and need to be provided.

def process_training_data(train_page_names):
    """Perform the training stage and return results in a dictionary.

    Params:
    train_page_names - list of training page names
    """
    print('Reading data')
    images_train = []
    labels_train = []
    for page_name in train_page_names:
        images_train = utils.load_char_images(page_name, images_train)
        labels_train = utils.load_labels(page_name, labels_train)
    labels_train = np.array(labels_train)

    print('Extracting features from training data')
    bbox_size = get_bounding_box_size(images_train)
    fvectors_train_full = images_to_feature_vectors(images_train, bbox_size)

    model_data = dict()
    model_data['labels_train'] = labels_train.tolist()
    model_data['bbox_size'] = bbox_size
    
    # PCA adapted from the labs
    # Calculating the principal components
    covx = np.cov(fvectors_train_full, rowvar=0)
    N = covx.shape[0]
    w, v = scipy.linalg.eigh(covx, eigvals=(N - 40, N - 1))
    v = np.fliplr(v)

    # Storing the eigenvectors in dictionary
    model_data['eigenvectors'] = v.tolist()

    print('Reducing to 10 dimensions')
    fvectors_train = reduce_dimensions(fvectors_train_full, model_data)

    model_data['fvectors_train'] = fvectors_train.tolist()
    return model_data


def load_test_page(page_name, model):
    """Load test data page.

    This function must return each character as a 10-d feature
    vector with the vectors stored as rows of a matrix.

    Params:
    page_name - name of page file
    model - dictionary storing data passed from training stage
    """
    bbox_size = model['bbox_size']
    images_test = utils.load_char_images(page_name)
    fvectors_test = images_to_feature_vectors(images_test, bbox_size)

    # Perform the dimensionality reduction.
    fvectors_test_reduced = reduce_dimensions(fvectors_test, model)
    return fvectors_test_reduced


def classify_page(page, model):
    """
    parameters:
    page - matrix, each row is a feature vector to be classified
    model - dictionary, stores the output of the training stage
    """

    """Performing nearest neighbour classification."""

    ##### Adapted Nearest Neighbour classification method from COM2004 lab sessions #####

    # Getting the training data
    fvectors_train = np.array(model['fvectors_train'])
    labels_train = np.array(model['labels_train'])

    # get sample range
    features=np.arange(0, fvectors_train.shape[1])

    # Making the page a numpy array to make it easier to work with
    page = np.array(page)

    # Select the desired features from the training and test data
    train = fvectors_train[:, features]
    test = page[:, features]

    # Super compact implementation of nearest neighbour
    x = np.dot(test, train.transpose())
    modtest=np.sqrt(np.sum(test * test, axis=1))
    modtrain=np.sqrt(np.sum(train * train, axis=1))
    dist = x / np.outer(modtest, modtrain.transpose()) # cosine distance
    nearest=np.argmax(dist, axis=1) # finding the index of nearest neighbour

    # Classified letter label
    label = labels_train[nearest]

    return label


def is_lowercase(char):
    # using regular expressions, I check to see if a char is lowercase or not
    return bool(re.match(r'[a-z]+$', char))

def correct_errors(page, labels, bboxes, model):
    """
    parameters:

    page - 2d array, each row is a feature vector to be classified
    labels - the output classification label for each feature vector
    bboxes - 2d array, each row gives the 4 bounding box coords of the character
    model - dictionary, stores the output of the training stage
    """

    # Getting the number of letters on page
    num_of_chars = bboxes.shape[0]

    # Initialising word and wordlist
    # characters that dont have spaces as subsequent characters are stored in 'word'
    # once a space has been detected it is stored in the wordlist
    word = ""
    wordlist = []

    for char in range(num_of_chars-1):
        currentChar = bboxes[char] # Current character
        nextChar = bboxes[char + 1] # Next character

        # Calculating the difference between the left sides of each character box
        startDiff = nextChar[2] - currentChar[2]
        # Calculating the difference between the right sides of each character box
        endDiff = nextChar[0] - currentChar[0]

        # Space and character tests
        # through trial and error, I found that the values 20 and 23 are enough to detect a space
        # new words are started if a non lowercase letter is detected as the next char
        if ((startDiff>20) and (endDiff >23)) or ((startDiff>23) and (endDiff>20)) or (is_lowercase(labels[char+1])==False):
            wordlist.append(word + labels[char]) # adding word to wordlist
            word = "" # Resetting the word variable
        else:
            word = word + labels[char] # adding another letter to the word

    #print(wordlist)
    # Example of wordlist: ['reason', 'I', 'don', ',t', 'see', 'why', 'your', 'parents', 'should', 'not',...]
    """
    Now that I have the list of words, I can check them in the words list provided in the brief
    If the word is present, those labels are correct
    If not, a similar word is searched for and will replace the unrecognised one
    """
    return labels
