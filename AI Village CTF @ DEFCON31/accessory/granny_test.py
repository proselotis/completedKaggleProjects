import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions
import numpy as np
from tqdm import tqdm

from art.estimators.classification import TensorFlowV2Classifier
from art.preprocessing.expectation_over_transformation import EoTImageRotationTensorFlow
from art.attacks.evasion import ProjectedGradientDescent
from tensorflow.keras.layers import Input


import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import scipy
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.applications.inception_v3 import InceptionV3

import imagenet_stubs
from imagenet_stubs.imagenet_2012_labels import label_to_name, name_to_label



def plot_prediction(img, probs, correct_class=None, target_class=None):

    # Initialize the subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))

    # Set the first plot to the input image
    fig.sca(ax1)
    ax1.imshow(img)

    # Determine the top ten labels and store them with their probabilities
    top_ten_indexes = list(probs[0].argsort()[-10:][::-1])
    top_probs = probs[0, top_ten_indexes]
    labels = [label_to_name(i) for i in top_ten_indexes]
    barlist = ax2.bar(range(10), top_probs)
    if target_class in top_ten_indexes:
        barlist[top_ten_indexes.index(target_class)].set_color('r')
    if correct_class in top_ten_indexes:
        barlist[top_ten_indexes.index(correct_class)].set_color('g')

    # Plot the probabilities and labels
    plt.sca(ax2)
    plt.ylim([0, 1.1])
    plt.xticks(range(10), labels, rotation='vertical')
    plt.ylabel("Probability")
    fig.subplots_adjust(bottom=0.2)
    plt.show()
     


# Load the MobileNetV2 model
model = MobileNetV2(weights='imagenet', include_top=True,
                    input_tensor=Input(shape=(224, 224, 3)))

loss = tf.keras.losses.CategoricalCrossentropy()

# Load and preprocess the original image
img_path = "ZL7JEZ4YO9ZX.jpg" #'GrannySmith_NYAS-Apples2.png'
img = image.load_img(img_path, target_size=(224,224)) #, target_size=(224, 224))

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

import tensorflow as tf
LABELS_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'
labels_path = tf.keras.utils.get_file('ImageNetLabels.txt', LABELS_URL)
labels = np.array(
    open(labels_path).read().splitlines()
)[1:]




eps = 10.0 #/ 255.0 # Attack budget for PGD
eps_step = .1 #/ 255.0 # Step size for PGD
num_steps = 100 # Number of iterations for PGD
y_target = np.array([948])
y = np.array([269])


## attempt 1
classifier = TensorFlowV2Classifier(model=model,
                                    nb_classes=1000,
                                    loss_object=loss,
                                    # preprocessing=preprocessing,
                                    # preprocessing_defences=None,
                                    clip_values=(0.0, 1.0),
                                    input_shape=(224, 224))

attack = ProjectedGradientDescent(estimator=classifier,
                                  eps=eps,
                                  max_iter=num_steps,
                                  eps_step=eps_step,
                                  targeted=True,
                                  verbose=True)

x_adv = attack.generate(x=x, y=y_target)
# eot_rotation = EoTImageRotationTensorFlow(nb_samples=10,
#                                           clip_values=(0.0, 1.0),
#                                           angles=22.5)   
# classifier_eot = TensorFlowV2Classifier(model=model,
#                                         nb_classes=1000,
#                                         loss_object=loss,
#                                         # preprocessing=preprocessing,
#                                         preprocessing_defences=[eot_rotation],
#                                         clip_values=(0.0, 1.0),
#                                         input_shape=(224, 224))
# attack_eot = ProjectedGradientDescent(estimator=classifier_eot,
#                                       eps=eps,
#                                       max_iter=num_steps,
#                                       eps_step=eps_step,
#                                       targeted=True)



# x_adv_eot = attack_eot.generate(x=x, y=y_target)

y_pred_adv = classifier.predict(x_adv)

img_adv = image.array_to_img(x_adv[0])
img_adv.save('art_granny_smith_pgd.png')

plot_prediction(np.squeeze(x_adv), y_pred_adv, correct_class=y, target_class=y_target)
