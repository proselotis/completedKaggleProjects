"""Baseline kernel for "Google Landmarks Recognition Challenge 2020".

Generates `submission.csv` in Kaggle format. When the number of training images
indicates that the kernel is being run against the public dataset,
simply copies `sample_submission.csv` to allow for quickly starting reruns
on the private dataset. When in a rerun against the private dataset,
makes predictions via retrieval, using DELG TensorFlow SavedModels for global
and local feature extraction.

First, ranks all training images by embedding similarity to each test image.
Then, performs geometric-verification and re-ranking on the `NUM_TO_RERANK`
most similar training images. For a given test image, each class' score is
the sum of the scores of re-ranked training images, and the predicted
class is the one with the highest aggregate score.

NOTE: For speed, this uses `pydegensac` as its RANSAC implementation.
Since the module has no interface for setting random seeds, RANSAC results
and submission scores will vary slightly between reruns.
"""

import copy
import csv
import gc
import operator
import os
import pathlib
import shutil

import pandas as pd
import numpy as np
import PIL
import pydegensac
from scipy import spatial
import tensorflow as tf
from pathlib import Path
import random

# Dataset parameters:
INPUT_DIR = os.path.join('..', 'input')

DATASET_DIR = os.path.join(INPUT_DIR, 'landmark-recognition-2020')
TEST_IMAGE_DIR = os.path.join(DATASET_DIR, 'test')
TRAIN_IMAGE_DIR = os.path.join(DATASET_DIR, 'train')
TRAIN_LABELMAP_PATH = os.path.join(DATASET_DIR, 'train.csv')

# DEBUGGING PARAMS:
NUM_PUBLIC_TRAIN_IMAGES = 1580470 # Used to detect if in session or re-run.
MAX_NUM_EMBEDDINGS = -1  # Set to > 1 to subsample dataset while debugging.

# Retrieval & re-ranking parameters:
NUM_TO_RERANK = 4
TOP_K = 4 #Number of retrieved images used to make prediction for a test image.

# RANSAC parameters:
MAX_INLIER_SCORE = 30
MAX_REPROJECTION_ERROR = 6.0
MAX_RANSAC_ITERATIONS = 100_000
HOMOGRAPHY_CONFIDENCE = 0.99

# Filtering parameters:
MAX_PHOTOS = 900
REQUIRED_PHOTOS = 5
RANDOM_SEED = 42
random.seed(RANDOM_SEED)


# DELG model:
SAVED_MODEL_DIR = '../input/delg-saved-models/local_and_global'
DELG_MODEL = tf.saved_model.load(SAVED_MODEL_DIR)
DELG_IMAGE_SCALES_TENSOR = tf.convert_to_tensor([0.70710677, 1.0, 1.4142135])
DELG_SCORE_THRESHOLD_TENSOR = tf.constant(175.)
DELG_INPUT_TENSOR_NAMES = [
    'input_image:0', 'input_scales:0', 'input_abs_thres:0'
]

# Global feature extraction:
NUM_EMBEDDING_DIMENSIONS = 2048
GLOBAL_FEATURE_EXTRACTION_FN = DELG_MODEL.prune(DELG_INPUT_TENSOR_NAMES,
                                                ['global_descriptors:0'])

# Local feature extraction:
LOCAL_FEATURE_NUM_TENSOR = tf.constant(1000)
LOCAL_FEATURE_EXTRACTION_FN = DELG_MODEL.prune(
    DELG_INPUT_TENSOR_NAMES + ['input_max_feature_num:0'],
    ['boxes:0', 'features:0'])


# Remove landmarks with absurd amount of photos
d = pd.read_csv(TRAIN_LABELMAP_PATH)
t = d.groupby("landmark_id").count()
t.columns = ["count"]
use = t[t["count"] < MAX_PHOTOS]
usablePhotos = set(d[d["landmark_id"].isin(use.index)]["id"].values)
# usablePhotos = set(d.groupby('landmark_id').head(MAX_PHOTOS)["id"].values)
# Create copies of photos with less than REQUIRED_PHOTOS
create = t[t["count"] < REQUIRED_PHOTOS]
create["needed"] = REQUIRED_PHOTOS - create["count"]
new_creations = create["needed"].to_dict()




def to_hex(image_id) -> str:
  return '{0:0{1}x}'.format(image_id, 16)


def get_image_path(subset, image_id):
  name = to_hex(image_id)
  return os.path.join(DATASET_DIR, subset, name[0], name[1], name[2],
                      '{}.jpg'.format(name))


def load_image_tensor(image_path,augment = False):
  if augment:
    ch = random.choices(["brightness","contrast","gamma","transpose","rotation","hue","channel"],
                      weights = [(1/7),(1/7),(1/7),(1/7),(1/7),(1/7),(1/7)])[0]
    if ch == "brightness":
      return tf.image.adjust_brightness(tf.convert_to_tensor(np.array(PIL.Image.open(image_path).convert('RGB'))), random.randint(-300, 300) /1000)
    elif ch == "contrast":
      return tf.image.adjust_contrast(tf.convert_to_tensor(np.array(PIL.Image.open(image_path).convert('RGB'))), random.randint(-300, 300) /1000)
    elif ch == "gamma": 
      return tf.image.adjust_gamma(tf.convert_to_tensor(np.array(PIL.Image.open(image_path).convert('RGB'))), random.randint(800, 1200) /1000)
    elif ch == "transpose":
      return tf.image.transpose(tf.convert_to_tensor(np.array(PIL.Image.open(image_path).convert('RGB'))))
    elif ch == "rotation":
      return tf.image.rot90(tf.convert_to_tensor(np.array(PIL.Image.open(image_path).convert('RGB'))))
    elif ch == "hue":
      return tf.image.adjust_hue(tf.convert_to_tensor(np.array(PIL.Image.open(image_path).convert('RGB'))), random.randint(-300, 300) /1000)
    elif ch == "channel":
      return tf.random.shuffle(tf.convert_to_tensor(np.array(PIL.Image.open(image_path).convert('RGB'))), seed=RANDOM_SEED)

         
  else:
    return tf.convert_to_tensor(np.array(PIL.Image.open(image_path).convert('RGB')))


def extract_global_features(image_root_dir,train = False):
  """Extracts embeddings for all the images in given `image_root_dir`."""

  if train:
    new_rows = [d[d["landmark_id"] == key].sample(new_creations[key],replace = True,random_state = RANDOM_SEED)["id"].values for key in new_creations.keys()]
    extras = [item for sublist in new_rows for item in sublist]
    image_paths = [x for x in pathlib.Path(image_root_dir).rglob('*.jpg') if str(x)[47:63] in usablePhotos] 
    new_image_paths = [Path(TRAIN_IMAGE_DIR + "/" + x[0] + "/" + x[1] + "/" + x[2] + "/" + x + ".jpg") for x in extras]
#     image_paths = [x for x in pathlib.Path(image_root_dir).rglob('*.jpg') if str(x)[47:63] in usablePhotos]
  else:
    image_paths = [x for x in pathlib.Path(image_root_dir).rglob('*.jpg')]
    new_image_paths = []
    
  num_embeddings = len(image_paths) + len(new_image_paths)
  if MAX_NUM_EMBEDDINGS > 0:
    num_embeddings = min(MAX_NUM_EMBEDDINGS, num_embeddings)

  ids = num_embeddings * [None]
  embeddings = np.empty((num_embeddings, NUM_EMBEDDING_DIMENSIONS))

  for i, image_path in enumerate(image_paths):
    if i >= len(image_paths):
      break

    ids[i] = int(image_path.name.split('.')[0], 16)
    image_tensor = load_image_tensor(image_path)
    features = GLOBAL_FEATURE_EXTRACTION_FN(image_tensor,
                                            DELG_IMAGE_SCALES_TENSOR,
                                            DELG_SCORE_THRESHOLD_TENSOR)
    embeddings[i, :] = tf.nn.l2_normalize(
        tf.reduce_sum(features[0], axis=0, name='sum_pooling'),
        axis=0,
        name='final_l2_normalization').numpy()
  if train:
    k = i + 1
    for j, image_path in enumerate(new_image_paths):
      i = j + k
      if i >= num_embeddings:
        break

      ids[i] = int(image_path.name.split('.')[0], 16)
      image_tensor = load_image_tensor(image_path,True)
      features = GLOBAL_FEATURE_EXTRACTION_FN(image_tensor,
                                            DELG_IMAGE_SCALES_TENSOR,
                                            DELG_SCORE_THRESHOLD_TENSOR)
      embeddings[i, :] = tf.nn.l2_normalize(
          tf.reduce_sum(features[0], axis=0, name='sum_pooling'),
          axis=0,
          name='final_l2_normalization').numpy()

  return ids, embeddings


def extract_local_features(image_path):
  """Extracts local features for the given `image_path`."""

  image_tensor = load_image_tensor(image_path)

  features = LOCAL_FEATURE_EXTRACTION_FN(image_tensor, DELG_IMAGE_SCALES_TENSOR,
                                         DELG_SCORE_THRESHOLD_TENSOR,
                                         LOCAL_FEATURE_NUM_TENSOR)

  # Shape: (N, 2)
  keypoints = tf.divide(
      tf.add(
          tf.gather(features[0], [0, 1], axis=1),
          tf.gather(features[0], [2, 3], axis=1)), 2.0).numpy()

  # Shape: (N, 128)
  descriptors = tf.nn.l2_normalize(
      features[1], axis=1, name='l2_normalization').numpy()

  return keypoints, descriptors


def get_putative_matching_keypoints(test_keypoints,
                                    test_descriptors,
                                    train_keypoints,
                                    train_descriptors,
                                    max_distance=0.9):
  """Finds matches from `test_descriptors` to KD-tree of `train_descriptors`."""

  train_descriptor_tree = spatial.cKDTree(train_descriptors)
  _, matches = train_descriptor_tree.query(
      test_descriptors,distance_upper_bound=max_distance)

  test_kp_count = test_keypoints.shape[0]
  train_kp_count = train_keypoints.shape[0]

  test_matching_keypoints = np.array([
      test_keypoints[i,]
      for i in range(test_kp_count)
      if matches[i] != train_kp_count
  ])
  train_matching_keypoints = np.array([
      train_keypoints[matches[i],]
      for i in range(test_kp_count)
      if matches[i] != train_kp_count
  ])

  return test_matching_keypoints, train_matching_keypoints


def get_num_inliers(test_keypoints, test_descriptors, train_keypoints,
                    train_descriptors):
  """Returns the number of RANSAC inliers."""

  test_match_kp, train_match_kp = get_putative_matching_keypoints(
      test_keypoints, test_descriptors, train_keypoints, train_descriptors)

  if test_match_kp.shape[
      0] <= 4:  # Min keypoints supported by `pydegensac.findHomography()`
    return 0

  try:
    _, mask = pydegensac.findHomography(test_match_kp, train_match_kp,
                                        MAX_REPROJECTION_ERROR,
                                        HOMOGRAPHY_CONFIDENCE,
                                        MAX_RANSAC_ITERATIONS)
  except np.linalg.LinAlgError:  # When det(H)=0, can't invert matrix.
    return 0

  return int(copy.deepcopy(mask).astype(np.float32).sum())


def get_total_score(num_inliers, global_score):
  local_score = min(num_inliers, MAX_INLIER_SCORE) / MAX_INLIER_SCORE
  return local_score #+ global_score


def rescore_and_rerank_by_num_inliers(test_image_id,
                                      train_ids_labels_and_scores):
  """Returns rescored and sorted training images by local feature extraction."""

  test_image_path = get_image_path('test', test_image_id)
  test_keypoints, test_descriptors = extract_local_features(test_image_path)

  for i in range(len(train_ids_labels_and_scores)):
    train_image_id, label, global_score = train_ids_labels_and_scores[i]

    train_image_path = get_image_path('train', train_image_id)
    train_keypoints, train_descriptors = extract_local_features(
        train_image_path)

    num_inliers = get_num_inliers(test_keypoints, test_descriptors,
                                  train_keypoints, train_descriptors)
    total_score = get_total_score(num_inliers, global_score)
    train_ids_labels_and_scores[i] = (train_image_id, label, total_score)

  train_ids_labels_and_scores.sort(key=lambda x: x[2], reverse=True)

  return train_ids_labels_and_scores


def load_labelmap():
  with open(TRAIN_LABELMAP_PATH, mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    labelmap = {row['id']: row['landmark_id'] for row in csv_reader}

  return labelmap


def get_prediction_map(test_ids, train_ids_labels_and_scores):
  """Makes dict from test ids and ranked training ids, labels, scores."""

  prediction_map = dict()

  for test_index, test_id in enumerate(test_ids):
    hex_test_id = to_hex(test_id)

    aggregate_scores = {}
    for _, label, score in train_ids_labels_and_scores[test_index][:TOP_K]:
      if label not in aggregate_scores:
        aggregate_scores[label] = 0
      aggregate_scores[label] += score

    label, score = max(aggregate_scores.items(), key=operator.itemgetter(1))

    prediction_map[hex_test_id] = {'score': score, 'class': label}

  return prediction_map


def get_predictions(labelmap):
  """Gets predictions using embedding similarity and local feature reranking."""

  test_ids, test_embeddings = extract_global_features(TEST_IMAGE_DIR)

  train_ids, train_embeddings = extract_global_features(TRAIN_IMAGE_DIR,True)

  train_ids_labels_and_scores = [None] * test_embeddings.shape[0]

  # Using (slow) for-loop, as distance matrix doesn't fit in memory.
  for test_index in range(test_embeddings.shape[0]):
    distances = spatial.distance.cdist(
        test_embeddings[np.newaxis, test_index, :], train_embeddings,
        'sqeuclidean')[0]
    partition = np.argpartition(distances, NUM_TO_RERANK)[:NUM_TO_RERANK]
    nearest = sorted([(train_ids[p], distances[p]) for p in partition],
                     key=lambda x: x[1])

    train_ids_labels_and_scores[test_index] = [
        (train_id, labelmap[to_hex(train_id)], 1. - cosine_distance)
        for train_id, cosine_distance in nearest
    ]

  del test_embeddings
  del train_embeddings
  del labelmap
  gc.collect()

  pre_verification_predictions = get_prediction_map(
      test_ids, train_ids_labels_and_scores)

#  return None, pre_verification_predictions

  for test_index, test_id in enumerate(test_ids):
    train_ids_labels_and_scores[test_index] = rescore_and_rerank_by_num_inliers(
        test_id, train_ids_labels_and_scores[test_index])

  post_verification_predictions = get_prediction_map(
      test_ids, train_ids_labels_and_scores)

  return pre_verification_predictions, post_verification_predictions


def save_submission_csv(predictions=None):
  """Saves optional `predictions` as submission.csv.

  The csv has columns {id, landmarks}. The landmarks column is a string
  containing the label and score for the id, separated by a ws delimeter.

  If `predictions` is `None` (default), submission.csv is copied from
  sample_submission.csv in `IMAGE_DIR`.

  Args:
    predictions: Optional dict of image ids to dicts with keys {class, score}.
  """

  if predictions is None:
    # Dummy submission!
    shutil.copyfile(
        os.path.join(DATASET_DIR, 'sample_submission.csv'), 'submission.csv')
    return

  with open('submission.csv', 'w') as submission_csv:
    csv_writer = csv.DictWriter(submission_csv, fieldnames=['id', 'landmarks'])
    csv_writer.writeheader()
    for image_id, prediction in predictions.items():
      label = prediction['class']
      score = prediction['score']
      csv_writer.writerow({'id': image_id, 'landmarks': f'{label} {score}'})


def main():
  labelmap = load_labelmap()
  num_training_images = len(labelmap.keys())
  print(f'Found {num_training_images} training images.')

  if num_training_images == NUM_PUBLIC_TRAIN_IMAGES:
    print(
        f'Found {NUM_PUBLIC_TRAIN_IMAGES} training images. Copying sample submission.'
    )
    save_submission_csv()
    return

  _, post_verification_predictions = get_predictions(labelmap)
  save_submission_csv(post_verification_predictions)


if __name__ == '__main__':
  main()