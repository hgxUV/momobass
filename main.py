from Nets import ThreePointFour, ThreePointFive, ThreePointSix, ThreePointTen
import numpy as np
from NN_helpers import draw_response3D, draw_response3D_pytorch
import torch.optim as optim
import torch.nn as nn
import torch
import matplotlib.pyplot as plt

task = 3.10

if task == 3.3:
    X = np.vstack([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.asarray([0, 0, 0, 1])
    draw_response3D(X, y, 'AND')

if task == 3.4:

    X = np.vstack([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.asarray([0, 0, 0, 1])

    X = X.astype('float32')
    y = y.astype('float32').reshape(-1, 1)

    X = torch.from_numpy(X)
    y = torch.from_numpy(y)

    net = ThreePointFour()

    #criterion = nn.CrossEntropyLoss()
    criterion = nn.L1Loss()
    #optimizer = optim.SGD(net.parameters(), lr=1e-5)
    optimizer = optim.Adam(net.parameters(), lr=1e-3)

    iterations = 3000
    losses = []

    for i in range(iterations):

        optimizer.zero_grad()
        outputs = net(X)
        loss = criterion(outputs, y)
        print('iter: {}, loss: {}'.format(i, loss))
        losses.append(loss)
        loss.backward()
        optimizer.step()

    fig, ax = plt.subplots()
    losses = np.asarray(losses)
    iter_vector = np.arange(iterations)
    ax.grid(which='both')
    ax.scatter(iter_vector, losses)
    plt.ylabel('wartość funkcji straty')
    plt.xlabel('liczba epok')
    plt.show()

    draw_response3D_pytorch(X, y, net)

if task == 3.6:

    X = np.vstack([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.asarray([0, 0, 0, 1])

    X = X.astype('float32')
    y = y.astype('float32').reshape(-1, 1)

    X = torch.from_numpy(X)
    y = torch.from_numpy(y)

    net = ThreePointSix()

    #criterion = nn.CrossEntropyLoss()
    criterion = nn.L1Loss()
    #optimizer = optim.SGD(net.parameters(), lr=1e-5)
    optimizer = optim.Adam(net.parameters(), lr=1e-3)

    iterations = 10000
    losses = []

    for i in range(iterations):

        optimizer.zero_grad()
        outputs = net(X)
        loss = criterion(outputs, y)
        print('iter: {}, loss: {}'.format(i, loss))
        losses.append(loss)
        loss.backward()
        optimizer.step()

    fig, ax = plt.subplots()
    losses = np.asarray(losses)
    iter_vector = np.arange(iterations)
    ax.grid(which='both')
    ax.scatter(iter_vector, losses)
    plt.ylabel('wartość funkcji straty')
    plt.xlabel('liczba epok')
    plt.show()

    draw_response3D_pytorch(X, y, net)


if task == 3.9:
    X = np.vstack([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.asarray([0, 1, 1, 0])
    draw_response3D(X, y, 'XOR')


if task == 3.10:

    X = np.vstack([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.asarray([0, 1, 1, 0])

    X = X.astype('float32')
    y = y.astype('float32').reshape(-1, 1)

    X = torch.from_numpy(X)
    y = torch.from_numpy(y)

    net = ThreePointTen()

    #criterion = nn.CrossEntropyLoss()
    criterion = nn.L1Loss()
    #optimizer = optim.SGD(net.parameters(), lr=1e-5)
    optimizer = optim.Adam(net.parameters(), lr=1e-3)

    iterations = 30000
    losses = []

    for i in range(iterations):

        optimizer.zero_grad()
        outputs = net(X)
        loss = criterion(outputs, y)
        print('iter: {}, loss: {}'.format(i, loss))
        losses.append(loss)
        loss.backward()
        optimizer.step()

    fig, ax = plt.subplots()
    losses = np.asarray(losses)
    iter_vector = np.arange(iterations)
    ax.grid(which='both')
    ax.scatter(iter_vector, losses)
    plt.ylabel('wartość funkcji straty')
    plt.xlabel('liczba epok')
    plt.show()

    draw_response3D_pytorch(X, y, net)
