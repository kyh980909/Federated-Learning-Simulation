import ast
import math
import wandb
from sklearn.model_selection import train_test_split
from collections import Counter
import random
from copy import copy
import numpy as np
from tensorflow.keras import datasets, layers, models, losses
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

wandb.init(project='Federated Learning', entity='yhkim')


def gap(weights):
    if len(weights.shape) == 4:
        row = np.mean(weights, axis=1)
        result = np.mean(row, axis=0)
    elif len(weights.shape) == 2:
        result = np.mean(weights, axis=0)
    else:
        raise print('가중치 잘못됨')

    return result


def split_ue_group(UE_weights, UE_NUM):
    layers = UE_weights[0].keys()
    concat_weight = {}
    concat_mean_weight = {}

    for layer in layers:
        total = np.zeros(
            (UE_NUM, UE_weights[0][layer][0].shape[-2:][0], UE_weights[0][layer][0].shape[-2:][1]))
        for i, UE in enumerate(UE_weights):
            total[i] = gap(UE[layer][0])
        concat_weight[layer] = total

    for layer in concat_weight.keys():
        concat_mean_weight[layer] = np.mean(concat_weight[layer], axis=0)

    UE_high_low = {}
    for layer in concat_weight.keys():
        true_cnt_list = []
        for x in range(UE_NUM):
            high_low = concat_weight[layer][x] > concat_mean_weight[layer]
            if len(high_low[high_low == True]) >= len(high_low[high_low == False]):
                true_cnt_list.append(True)
            else:
                true_cnt_list.append(False)
        UE_high_low[layer] = true_cnt_list

    result = [0 for _ in range(UE_NUM)]
    for layer in UE_high_low.keys():
        for i, x in enumerate(UE_high_low[layer]):
            if x == True:
                result[i] += 1

    high_ue_list = []
    low_ue_list = []

    for i, x in enumerate(result):
        if x >= math.ceil(len(UE_high_low.keys()) / 2):
            high_ue_list.append(i)
        else:
            low_ue_list.append(i)

    return (high_ue_list, low_ue_list)


(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
idx = np.argsort(y_test)
x_train_sorted = x_test[idx]
y_train_sorted = y_test[idx]

UE_NUM = 10  # UE 개수

UE = []
for _ in range(UE_NUM):
    UE.append({"x_train": [], "y_train": []})

random.seed(45)

total = 0
random_num_list = []
for _ in range(UE_NUM):
    random_num = random.randrange(1000, 12000)
    total += random_num
    random_num_list.append(random_num)

x_eval_dataset = x_train[total:]
y_eval_dataset = y_train[total:]

start = 0
for i in range(UE_NUM):
    if i == 0:
        UE[i]['x_train'] = x_train[:random_num_list[i]]
        UE[i]['y_train'] = y_train[:random_num_list[i]]
    else:
        UE[i]['x_train'] = x_train[start:start+random_num_list[i]]
        UE[i]['y_train'] = y_train[start:start+random_num_list[i]]
    start += random_num_list[i]


x_train, x_test, y_train, y_test = [], [], [], []
for i in range(UE_NUM):
    x_train_temp, x_test_temp, y_train_temp, y_test_temp = train_test_split(
        UE[i]['x_train'], UE[i]['y_train'], test_size=0.2, random_state=45)
    x_train.append(x_train_temp)
    x_test.append(x_test_temp)
    y_train.append(y_train_temp)
    y_test.append(y_test_temp)


with open('high_group_list.txt', 'r') as f:
    high_ue_list = ast.literal_eval(f.read())

high_group_global_accuracy = []  # 각 레이어 들의 가중치가 평균보다 높은 UE 그룹의 정확도
high_group_global_loss = []

for round in range(100):  # Communication Round, Global epoch
    high_ue_learning_result_list = []

    # High Group
    for i in high_ue_list:
        if round == 0:
            model = tf.keras.models.load_model('fl_model_gap')
            model.compile(optimizer='SGD',
                          loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            high_ue_learning_result_list.append(model.fit(
                x_train[i], y_train[i], batch_size=100, epochs=1, validation_data=(x_test[i], y_test[i])))
            tf.keras.backend.clear_session()
        else:
            model = tf.keras.models.load_model('high_group')
            model.compile(optimizer='SGD',
                          loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            high_ue_learning_result_list.append(model.fit(
                x_train[i], y_train[i], batch_size=100, epochs=1, validation_data=(x_test[i], y_test[i])))
            tf.keras.backend.clear_session()

    UE_weights = []

    for model in high_ue_learning_result_list:
        layer_weights = {}
        for x in model.model.layers:
            if len(x.get_weights()) > 0:
                layer_weights[x.name] = x.get_weights()
        UE_weights.append(layer_weights)

    server_model = models.Sequential()
    server_model.add(layers.Conv2D(filters=6, kernel_size=(
        5, 5), strides=1, activation='tanh', input_shape=(32, 32, 1)))
    server_model.add(layers.AveragePooling2D(pool_size=2, strides=2))
    server_model.add(layers.Conv2D(
        filters=16, kernel_size=(5, 5), strides=1, activation='tanh'))
    server_model.add(layers.AveragePooling2D(pool_size=2, strides=2))
    server_model.add(layers.Flatten())
    server_model.add(layers.Dense(120, activation='tanh'))
    server_model.add(layers.Dense(84, activation='tanh'))
    server_model.add(layers.Dense(10, activation='softmax'))
    tf.keras.backend.clear_session()

    sum_weights = {}

    for i in range(len(list(UE_weights[0].keys()))):
        weight_shape = [0]
        bias_shape = [0]
        for dim in UE_weights[0][list(UE_weights[0].keys())[i]][0].shape:
            weight_shape.append(dim)
        for dim in UE_weights[0][list(UE_weights[0].keys())[i]][1].shape:
            bias_shape.append(dim)
        sum_weights.update({list(UE_weights[0].keys())[i]: {
            'weight': np.empty(weight_shape), 'bias': np.empty(bias_shape)}})

        for UE in UE_weights:
            sum_weights[list(UE.keys())[i]]['weight'] = np.append(
                sum_weights[list(UE.keys())[i]]['weight'], [UE[list(UE.keys())[i]][0]], axis=0)
            sum_weights[list(UE.keys())[i]]['bias'] = np.append(
                sum_weights[list(UE.keys())[i]]['bias'], [UE[list(UE.keys())[i]][1]], axis=0)

    for layer in sum_weights.keys():
        for model_layer in server_model.layers:
            if layer == model_layer.name:
                model_layer.set_weights([np.mean(sum_weights[layer]['weight'], axis=0), np.mean(
                    sum_weights[layer]['bias'], axis=0)])

    server_model.compile(
        optimizer='SGD', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    result = server_model.evaluate(
        x=x_eval_dataset, y=y_eval_dataset, batch_size=128)
    print(f'Round {round} -- test loss, test acc: {result}')

    server_model.save('high_group')
    high_group_global_loss.append(result[0])
    high_group_global_accuracy.append(result[1])

    wandb.log({'high group global accuracy': high_group_global_accuracy[round],
               'high group global loss': high_group_global_loss[round], 'global epoch': round})
