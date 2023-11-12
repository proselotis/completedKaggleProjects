from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from IPython.display import clear_output
from matplotlib import pyplot as plt

from art.estimators.classification import BlackBoxClassifier
from art.attacks.evasion import HopSkipJump
from art.utils import to_categorical
from art.utils import load_dataset, get_file, compute_accuracy
from PIL import Image
import base64
import requests
import pandas as pd
import time 
####################


import tensorflow as tf
LABELS_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'
labels_path = tf.keras.utils.get_file('ImageNetLabels.txt', LABELS_URL)
labels = np.array(
    open(labels_path).read().splitlines()
)[1:]
def query(input_data):
    try:
        response = requests.post('http://granny.advml.com/score', json={'data': input_data})
        res = response.json()
    except:
        try:
            time.sleep(5)
            response = requests.post('http://granny.advml.com/score', json={'data': input_data})
            res = response.json()
        except:
            try:
                time.sleep(60)
                response = requests.post('http://granny.advml.com/score', json={'data': input_data})
                res = response.json()
            except:
                try:
                    time.sleep(120)
                    response = requests.post('http://granny.advml.com/score', json={'data': input_data})
                    res = response.json()
                except:
                    try:
                        time.sleep(300)
                        response = requests.post('http://granny.advml.com/score', json={'data': input_data})
                        res = response.json()
                    except:
                        print("too many failures")
    return res

def get_prediction(image,target_label=None):
    import time
    saveable_image = Image.fromarray(image.astype(np.uint8))
    file_name = "art_123" + ".png"
    saveable_image.save(file_name)
    with open(file_name, 'rb') as f:
        input_data = base64.b64encode(f.read()).decode()
    res = query(input_data)
    if 'message' not in res.keys():
        try:
            df = pd.DataFrame([list(x) for x in list({v:k for k,v in dict(res['output']).items()}.items())])
        except:
            print(res)
            df = pd.DataFrame([[1,2]],columns=[0,1])
    else:
        df = pd.DataFrame([[1,2]],columns=[0,1])
    if len(df) != 1000:
        missing = set(labels) - set(df[0].tolist())
        for idx,miss in enumerate(missing): 
            df.loc[-(1 + idx)] = [miss,0.0]
    predictions = df.set_index(0).loc[labels][1].tolist()
    if target_label == None:
        return predictions
    else:
        return preditcions[target_label]


####################

image_path = 'timber_wolf.png' 
original_image = Image.open(image_path).resize((264,264)).convert("L")
original_array = np.array(original_image)

####################
classifier = BlackBoxClassifier(get_prediction, input_shape=original_array.shape,
                                nb_classes=1000, clip_values=(0, 255)
                                )
print('Prediction from API is: ' + str(np.argmax(classifier.predict(original_array.astype(np.float32)), axis=1)[0]))

####################

# Generate HopSkipJump attack against black box classifier
attack = HopSkipJump(classifier=classifier, targeted=True,
                     max_iter=10000, max_eval=10000, init_eval=1000,
                     batch_size=1,init_size=1000,verbose=True)

x_adv = np.array(original_array.astype(np.float32))
y_target = np.zeros(1000)
y_target[948] = 1

# tar_image_path = 'best(4).png' 
# tar_original_image = Image.open(tar_image_path).resize((264,264))
# target_array = np.array(np.array(tar_original_image).astype(np.float32))
x_adv = attack.generate(x=x_adv, y=y_target)
    
    # print("Adversarial image at step %d." % (i * iter_step), "L2 error", 
    #       np.linalg.norm(np.reshape(x_adv[0] - target_image, [-1])),
    #       "and class label %d." % np.argmax(classifier.predict(x_adv)[0]))
adversarial_image_cast = Image.fromarray(x_adv.astype(np.uint8))
adversarial_image_cast.save(f'OPT_best_checkpoint_0.png') 

    
