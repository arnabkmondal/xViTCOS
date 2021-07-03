import tensorflow as tf
import tensorflow_addons as tfa
from vit_keras import vit, utils, visualize
from dataset import COVIDxCTDataset
from math import ceil
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt

Training = True
DATA_DIR = '../2A_images/'
# TRAIN_LABEL_FILE = '../train_COVIDx_CT-2A.txt'
TRAIN_LABEL_FILE = './resampled_train_COVIDx_CT-2A.txt'
VAL_LABEL_FILE = '../val_COVIDx_CT-2A.txt'
TEST_LABEL_FILE = '../test_COVIDx_CT-2A.txt'
BATCH_SIZE = 64
IMAGE_SIZE = (224, 224)
PATCH_SIZE = 16
NUM_EPOCHS = 20

INPUT_HEIGHT = 512
INPUT_WIDTH = 512
MAX_BBOX_JITTER = 0.075
MAX_ROTATION = 15
MAX_SHEAR = 0.2
MAX_PIXEL_SHIFT = 15
MAX_PIXEL_SCALE_CHANGE = 0.15
CLASS_NAMES = ['Normal', 'Pneumonia', 'COVID-19']
N_CLASS = len(CLASS_NAMES)
CLASS_DICT = {0: 'Normal', 1: 'Pneumonia', 2: 'COVID-19'}

dataset = COVIDxCTDataset(
    DATA_DIR,
    image_height=INPUT_HEIGHT,
    image_width=INPUT_WIDTH,
    target_height=IMAGE_SIZE[0],
    target_width=IMAGE_SIZE[1],
    max_bbox_jitter=MAX_BBOX_JITTER,
    max_rotation=MAX_ROTATION,
    max_shear=MAX_SHEAR,
    max_pixel_shift=MAX_PIXEL_SHIFT,
    max_pixel_scale_change=MAX_PIXEL_SCALE_CHANGE
)

tr_dataset, tr_num_images, tr_batch_size = dataset.train_dataset(TRAIN_LABEL_FILE, BATCH_SIZE)
tr_iter_per_epoch = ceil(tr_num_images / tr_batch_size)

val_dataset, val_num_images, val_batch_size = dataset.validation_dataset(VAL_LABEL_FILE, BATCH_SIZE)
val_iter_per_epoch = ceil(val_num_images / val_batch_size)

test_dataset, test_num_images, test_batch_size = dataset.validation_dataset(TEST_LABEL_FILE, BATCH_SIZE)
test_iter_per_epoch = ceil(test_num_images / test_batch_size)

base_model = vit.vit_b16(
    image_size=IMAGE_SIZE[0],
    activation='softmax',
    pretrained=True,
    include_top=False,
    pretrained_top=False
)

base_model.summary()

x = base_model.output
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(256, activation='relu')(x)
y = tf.keras.layers.Dense(N_CLASS, activation='softmax')(x)
model = tf.keras.Model(base_model.input, y)
model.summary()
optimizer = tf.keras.optimizers.Adam(
    learning_rate=tf.keras.experimental.CosineDecay(
        initial_learning_rate=1e-4, decay_steps=5000))
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_acc')
val_loss = tf.keras.metrics.Mean(name='val_loss')
val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_acc')


@tf.function
def train_step(data):
    with tf.GradientTape() as tape:
        inp = data['image']
        y_true = data['label']
        y_pred = model(inp)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()(y_true, y_pred)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(y_true, y_pred)


@tf.function
def val_step(data):
    inp = data['image']
    y_true = data['label']
    y_pred = model(inp, training=False)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()(y_true, y_pred)
    val_loss(loss)
    val_accuracy(y_true, y_pred)


@tf.function
def predict_step(data):
    img = data['image']
    y_true = data['label']
    y_hat = model(img, training=False)
    return y_true, y_hat


ckpt_path = './ViT/checkpoint/'
result_path = './ViT/results/'
BEST_VAL_LOSS = 999999
BEST_VAL_ACC = 0
current_patience = patience = 3

if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path)
if not os.path.exists(result_path):
    os.makedirs(result_path)

tr_data_iter = iter(tr_dataset)

for step in range(NUM_EPOCHS * tr_iter_per_epoch):
    tr_data = next(tr_data_iter)
    train_step(tr_data)
    if step % 100 == 0:
        for idx, data in enumerate(val_dataset):
            val_step(data)
        print(
            f'\nStep {step + 1}, '
            f'Loss: {train_loss.result().numpy()}, '
            f'Accuracy: {train_accuracy.result().numpy() * 100}, '
            f'Val Loss: {val_loss.result().numpy()}, '
            f'Val Accuracy: {val_accuracy.result().numpy() * 100}\n'
        )

        print(f'\n LR: {tf.keras.backend.get_value(optimizer.learning_rate)}\n')

        if BEST_VAL_ACC < val_accuracy.result().numpy():
            print('Improved')
            BEST_VAL_ACC = val_accuracy.result().numpy()
            model.save_weights(f'{ckpt_path}covid_vit.h5')
            current_patience = patience
        else:
            current_patience -= 1
            if current_patience == 0:
                # tf.keras.backend.set_value(optimizer.learning_rate,
                #                            max(tf.keras.backend.get_value(optimizer.learning_rate) * 0.5, 1e-7))
                # print(f'\n LR: {tf.keras.backend.get_value(optimizer.learning_rate)}\n')
                current_patience = patience

        train_loss.reset_states()
        train_accuracy.reset_states()
        val_loss.reset_states()
        val_accuracy.reset_states()

        plt.close('all')
        fig, axes = plt.subplots(3, 3, figsize=(16, 16))
        indices = np.random.choice(list(range(BATCH_SIZE)), 9)
        for index, ax in zip(indices, axes.ravel()):
            image = tr_data['image'].numpy()[index]
            # Display
            cls = tr_data['label'].numpy()[index]
            ax.imshow(image)
            ax.set_title('Class: {} ({})'.format(CLASS_NAMES[cls], cls))
        plt.savefig(f'{result_path}test_img_{step}.png')

model.load_weights(f'{ckpt_path}covid_vit.h5')

y_pred_test = []
test_labels = []
for idx, data in enumerate(test_dataset):
    test_y, pred_y = predict_step(data)
    y_pred_test.append(pred_y.numpy())
    test_labels.append(test_y.numpy())

y_pred_test = np.concatenate(y_pred_test, axis=0)
test_labels = np.concatenate(test_labels, axis=0)

y_pred_test = np.concatenate(y_pred_test, axis=0)
test_labels = np.concatenate(test_labels, axis=0)
prediction_dict = dict()
prediction_dict['test_y_pred'] = y_pred_test
prediction_dict['test_y_pred_cat'] = np.argmax(y_pred_test, axis=1)
prediction_dict['test_y'] = test_labels

with open(f'{result_path}pred_true.pkl', 'wb') as f:
    pickle.dump(prediction_dict, f)
