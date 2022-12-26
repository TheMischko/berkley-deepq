import numpy as np
import torch
from matplotlib import pyplot as plt

from Layer import Layer
from Loader import Loader


def train(model, crit, loader, optimizer, stats, subset_name='train'):
    """
    Trenovani dopredneho modelu metodou minibatch SGD

    vstup:
        model       ... objekt implementujici metody `backward` a `forward`
        crit        ... kriterium, vraci loss (skalar); musi byt objekt implementujici metody `forward` a `backward`
        loader      ... objekt tridy `BatchLoader`, ktery prochazi data po davkach
        optimizer   ... objekt updatujici parametry modelu a implmentujici metodu `step`
        stats       ... objekt typu Stats
    """
    step = 0
    for X_batch, Y_batch in loader.__iter__():

        # forward
        score = model.forward(X_batch)

        # loss
        loss = crit.forward(score, Y_batch)

        # backward
        dscore, _ = crit.backward()
        _, dparams = model.backward(dscore)

        # update parametru
        optimizer.step(dparams)

        # accuracy evaluation
        _, pred = score.max(dim=1)
        accuracy = torch.sum(pred == Y_batch).float() / X_batch.shape[0]
        stats.append_batch_stats(subset_name, loss=float(loss), acc=float(accuracy))
        print_evaluation(loss, accuracy, step, subset_name)
        step += 1


def validate(model: Layer, crit, loader: Loader, stats, subset_name="valid"):
    """
    Model validation

    input:
        model   ... object of class Layer; must implement methods forward and backward
        crit    ... criteria, returning loss (scalar value); must implement methods forward and backward
        loader  ... object of class Loader, returning data in batches
    """

    model.eval()
    # device = next(model.parameters()).device
    step = 0
    for X_batch, Y_batch in loader.__iter__():
        # send data to right computation device
        # X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

        # forward
        score = model.forward(X_batch)

        # loss
        loss = crit.forward(score, Y_batch)

        #accuracy evaluation
        _, prediction = score.max(dim=1)
        accuracy = torch.sum(prediction == Y_batch, dtype=np.float) / X_batch.shape[0]
        stats.append_batch_stats(subset_name, loss=float(loss), acc=float(accuracy))
        print_evaluation(loss, accuracy, step, subset_name)
        step += 1


def eval_numerical_gradient(f, x, df, h=0.00001):
    """
    Computes numerical gradient of function `f`.
    input:
        f   ... function that takes single argument
        x   ... numpy array representing point to evaluate gradient at
    """
    if h is None:
        h = max(1e-6, 0.1 * x.std())

    x_ = x.flatten()  # sdili s `x` pamet
    dx = torch.zeros_like(x_)

    for i in range(x_.shape[0]):
        v = float(x_[i])  # musi byt obaleno `float`, jinak vraci `Tensor`, ktery sdili pamet
        x_[i] = v + h
        p = f(x)

        x_[i] = v - h
        n = f(x)
        x_[i] = v

        dx[i] = torch.sum((p - n) * df) / (2. * h)
    return dx.reshape(x.shape)


def rel_error(x, y):
    return float(torch.max(torch.abs(x - y) / (torch.clamp(torch.abs(x) + torch.abs(y), min=1e-8))))


def check_gradients(model: Layer, inputs, d_outputs, input_names=None, h=0.00001):
    """
    Function that tries forward and backward run for each parameter of given model
    and check gradients against numerical difference

    inputs:
        model       ... instance of class Layer implementing methods `backward` and `forward`
        inputs      ... `tuple` of input layer
        d_outputs   ... N x H gradient matrix on the output of network
        input_names ... `tuple` with names of inputs
    returns:
        grads       ... gradients counted with `model.backward(d_outputs)`
        grad_nums   ... gradients counted with numerical method
    """

    # analytical gradients
    out = model.forward(inputs)
    dinputs, dparams = model.backward(d_outputs)
    grads = {'inputs': dinputs, **dparams}

    # numericky gradient (diference) na vstup (je vzdy az na toleranci spravne)
    grads_num = {}

    # vrstva muze mit libovolny pocet vstupu
    if not isinstance(inputs, tuple):
        inputs = (inputs,)
    if input_names is None:
        input_names = tuple(f'input{i + 1}' for i in range(len(inputs)))

    # checkni gradient pro kazdy pro kazdy vstup
    for i, x in enumerate(inputs):
        # gradient pocitame, pouze pokud jsou vstupem realna cisla (ne integery apod.)
        if x.dtype in (torch.float16, torch.float32, torch.float64):
            grads_num['inputs'] = eval_numerical_gradient(lambda _: model.forward(*inputs), x, d_outputs, h=h)
            print(f'd{input_names[i]} error: ', rel_error(grads['inputs'], grads_num['inputs']))

    # numericky gradient na parametry modelu
    for name in model.params:
        grads_num[name] = eval_numerical_gradient(lambda _: model.forward(*inputs), model.params[name], d_outputs,
                                                         h=h)
        print(f'd{name} error: ', rel_error(grads[name], grads_num[name]))

    return grads, grads_num


def predict_and_show(rgb, model, transform, classes=None):
    """
    inputs:
        rgb         ... numpy.ndarray with dimension width x height x color channels of type np.uint8
        model       ... object of type Module
        transform   ... image preprocessing
        classes     ... list of target classes
    """
    model.eval()

    # preprocess
    x = transform(rgb)
    x = x[None]

    # forward
    score = model(x)
    prob = softmax(score)

    if classes is None:
        classes = [str(i) for i in range(prob.shape[0])]

        # vykresleni matplotlib
    plt.figure(figsize=(5, 5))
    plt.imshow(np.array(rgb))
    ids = np.argsort(-score)
    for i, ci in enumerate(ids[:10]):
        text = '{:>5.2f} %  {}'.format(100. * prob[ci], classes[ci])
        if len(text) > 40:
            text = text[:40] + '...'
        plt.gcf().text(1., 0.8 - 0.075 * i, text, fontsize=24)
    plt.subplots_adjust()

def softmax(x):
    return np.exp(x)/sum(np.exp(x))


def print_evaluation(loss, accuracy, step, subset_name):
    print("{:} data stats for {:} step".format(subset_name, step))
    print("Loss = {:.3f}".format(loss))
    print("Accuracy = {:.3f}".format(accuracy))
    print("------------------------------")