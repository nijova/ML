# https://aayushmnit.github.io/posts/2018/06/Building_neural_network_from_scratch/

def softmax_crossentropy_with_logits(logits, reference_answers):
    logits_for_answers = logits[np.arange(len(logits)),reference_answers]
    xentropy = - logits_for_answers + np.log(np.sum(np.exp(logits),axis=-1))
    return xentropy

def grad_softmax_crossentropy_with_logits(logits, reference_answers):
    ones_for_answers = np.zeros_like(logits)
    ones_for_answers[np.arange(len(logits)),reference_answers] = 1
    softmax = np.exp(logits) / np.exp(logits).sum(axis=-1,keepdims=True)
    return (- ones_for_answers + softmax) / logits.shape[0]


network = []
network.append(Dense(X_train.shape[1],100))
network.append(ReLU())
network.append(Dense(100,10))


def forward(network, inp):
    activations = []
    for layer in network:
        activations.append(layer.forward(inp))
        inp = activations[-1]
    return activations


def predict(network,X):
    logits = forward(network,X)[-1]
    return logits.argmax(axis=-1)


def train(network,X,y):
    layer_activations = forward(network,X)
    layer_inputs = [X]+layer_activations
    logits = layer_activations[-1]

    loss = softmax_crossentropy_with_logits(logits,y)
    loss_grad = grad_softmax_crossentropy_with_logits(logits,y)

    for layer_index in range(len(network))[::-1]:
        layer = network[layer_index]
        loss_grad = layer.backward(layer_inputs[layer_index],loss_grad)
        
    return np.mean(loss)


from tqdm import trange
def iterate_minibatches(inputs, targets, batchsize):
    indices = np.random.permutation(len(inputs))
    for start_idx in trange(0, len(inputs) - batchsize + 1, batchsize):
        excerpt = indices[start_idx:start_idx + batchsize]
        yield inputs[excerpt], targets[excerpt]


for epoch in range(25):
    for x_batch,y_batch in iterate_minibatches(X_train,y_train,batchsize=32):
        train(network,x_batch,y_batch)
