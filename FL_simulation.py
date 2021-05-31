import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models, losses
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import Counter
from sklearn.model_selection import train_test_split
import os
import wandb

wandb.init(project='Federated Learning', entity='yhkim')

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

# mnist 학습, 성능 검증 데이터셋 나누기
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
idx = np.argsort(y_test)
x_train_sorted = x_test[idx]
y_train_sorted = y_test[idx]
x_eval_dataset = x_train[29854:]
y_eval_dataset = y_train[29854:]

# 균형 데이터셋 5세트
UE = []
for _ in range(5):
    UE.append({"x_train": [], "y_train": []})

random.seed(45)

total = 0
random_num_list = []
for _ in range(5):
    random_num = random.randrange(1000, 12000)
    total += random_num
    random_num_list.append(random_num)

print('사용한 데이터 수 :', total)
print(random_num_list)

start = 0
for i in range(5):
    if i == 0:
        UE[i]['x_train'] = x_train[:random_num_list[i]]
        UE[i]['y_train'] = y_train[:random_num_list[i]]
    else:
        UE[i]['x_train'] = x_train[start:start+random_num_list[i]]
        UE[i]['y_train'] = y_train[start:start+random_num_list[i]]
    start += random_num_list[i]

# 불균형 데이터 5개 세트
UE.append({'x_train': x_test[y_test == 0],
           'y_train': y_test[y_test == 0]})  # UE6

x = x_test[y_test == 1]
y = y_test[y_test == 1]
x = np.append(x, x_test[y_test == 3], axis=0)
y = np.append(y, y_test[y_test == 3], axis=0)

UE.append({'x_train': x, 'y_train': y})  # UE7

x = x_test[y_test == 2]
y = y_test[y_test == 2]
x = np.append(x, x_test[y_test == 4], axis=0)
y = np.append(y, y_test[y_test == 4], axis=0)
x = np.append(x, x_test[y_test == 9], axis=0)
y = np.append(y, y_test[y_test == 9], axis=0)

UE.append({'x_train': x, 'y_train': y})  # UE8

x = x_test[y_test == 5]
y = y_test[y_test == 5]
x = np.append(x, x_test[y_test == 6], axis=0)
y = np.append(y, y_test[y_test == 6], axis=0)

UE.append({'x_train': x, 'y_train': y})  # UE9

x = x_test[y_test == 8]
y = y_test[y_test == 8]
x = np.append(x, x_test[y_test == 7], axis=0)
y = np.append(y, y_test[y_test == 7], axis=0)

UE.append({'x_train': x, 'y_train': y})  # UE10


x_train, x_test, y_train, y_test = [], [], [], []
for i in range(10):
    x_train_temp, x_test_temp, y_train_temp, y_test_temp = train_test_split(
        UE[i]['x_train'], UE[i]['y_train'], test_size=0.2, random_state=45)
    x_train.append(x_train_temp)
    x_test.append(x_test_temp)
    y_train.append(y_train_temp)
    y_test.append(y_test_temp)

global_accuracy = []  # 10개의 UE가 각각 학습한 가중치를 평균내어 산출한 정확도
global_loss = []  # 10개의 UE가 각각 학습한 가중치를 평균내어 산출한 loss
for round in range(100):  # Communication Round, Global epoch
    # 각 UE 학습
    learning_result_list = []
    for i in range(10):
        if os.path.isdir('fl_model'):
            model = tf.keras.models.load_model('fl_model')
            model.compile(optimizer='SGD',
                          loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            learning_result_list.append(model.fit(
                x_train[i], y_train[i], batch_size=100, epochs=1, validation_data=(x_test[i], y_test[i])))
            tf.keras.backend.clear_session()
        else:
            model = models.Sequential()
            model.add(layers.Conv2D(filters=6, kernel_size=(5, 5),
                                    strides=1, activation='tanh', input_shape=(32, 32, 1)))
            model.add(layers.AveragePooling2D(pool_size=2, strides=2))
            model.add(layers.Conv2D(filters=16, kernel_size=(
                5, 5), strides=1, activation='tanh'))
            model.add(layers.AveragePooling2D(pool_size=2, strides=2))
            model.add(layers.Flatten())
            model.add(layers.Dense(120, activation='tanh'))
            model.add(layers.Dense(84, activation='tanh'))
            model.add(layers.Dense(10, activation='softmax'))
            model.compile(optimizer='SGD',
                          loss='sparse_categorical_crossentropy', metrics=['accuracy'])  # lr = 0.01
            learning_result_list.append(model.fit(
                x_train[i], y_train[i], batch_size=100, epochs=1, validation_data=(x_test[i], y_test[i])))
            tf.keras.backend.clear_session()

    # 각 UE 학습 리스트 안에 딕셔너리로 저장
    UE_weights = []

    for model in learning_result_list:
        layer_weights = {}
        for x in model.model.layers:
            if len(x.get_weights()) > 0:
                layer_weights[x.name] = x.get_weights()
        UE_weights.append(layer_weights)

    # 서버 모델 생성
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

    # 10개 UE가 각자 학습한 가중치 취합
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

    # 서버 모델에 가중치 설정하는 코드
    for layer in sum_weights.keys():
        for model_layer in server_model.layers:
            if layer == model_layer.name:
                model_layer.set_weights([np.mean(sum_weights[layer]['weight'], axis=0), np.mean(  # 각 UE 가중치 평균값으로 FL
                    sum_weights[layer]['bias'], axis=0)])

    # FL 성능 검증
    server_model.compile(
        optimizer='SGD', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    result = server_model.evaluate(
        x=x_eval_dataset, y=y_eval_dataset, batch_size=100)
    print('test loss, test acc:', result)
    server_model.save('fl_model')
    global_loss.append(result[0])
    global_accuracy.append(result[1])
    wandb.log(
        {'global accuracy': result[1], 'global loss': result[0], 'global epoch': round+1})

with open('global_accuracy_result.txt', 'w') as f:
    for x in global_accuracy:
        f.write(x+'\n')

with open('global_loss_result.txt', 'w') as f:
    for x in global_loss:
        f.write(x+'\n')
