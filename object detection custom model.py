#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install -q tflite-model-maker')
get_ipython().system('pip install -q pycocotools')
get_ipython().system('pip install -q tflite-support')


# In[ ]:


import numpy as np
import os

from tflite_model_maker.config import ExportFormat
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

import tensorflow as tf
assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)


# In[ ]:


spec = model_spec.get('efficientdet_lite1')


# In[ ]:


train_data, validation_data, test_data = object_detector.DataLoader.from_csv('/content/drive/MyDrive/training/train.csv')


# In[ ]:


model = object_detector.create(train_data, model_spec=spec, batch_size=32, train_whole_model=True, validation_data=validation_data)


# In[ ]:


model.export(export_dir='image_dir/mytflite')

