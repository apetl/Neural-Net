import numpy as np

class neuronLayer():
    def __init__(self, neurons, inputs):
        self.synapticWeights = 2 * np.random.random((inputs, neurons)) - 1

class neuralNet:
    def __init__(self, layer1, layer2):
        self.layer1 = layer1
        self.layer2 = layer2
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoidDer(self, x):
        return x * (1 - x)
    
    def train(self, tsi, tso, epochs):
        for i in range(epochs):
            layerOut1, layerOut2 = self.forward(tsi)

            layerErr2 = tso - layerOut2
            layerDel2 = layerErr2 * self.sigmoidDer(layerOut2)

            layerErr1 = layerDel2.dot(self.layer2.synapticWeights.T)
            layerDel1 = layerErr1 * self.sigmoidDer(layerOut1)

            layerAdj1 = tsi.T.dot(layerDel1)
            layerAdj2 = layerOut1.T.dot(layerDel2)

            self.layer1.synapticWeights += layerAdj1
            self.layer2.synapticWeights += layerAdj2

    def forward(self, inputs):
        layerOut1 = self.sigmoid(np.dot(inputs, self.layer1.synapticWeights))
        layerOut2 = self.sigmoid(np.dot(layerOut1, self.layer2.synapticWeights))
        return layerOut1, layerOut2

    def test(self, inputs):
        layerOut1 = self.sigmoid(np.dot(inputs, self.layer1.synapticWeights))
        layerOut2 = self.sigmoid(np.dot(layerOut1, self.layer2.synapticWeights))
        return layerOut2

    def weightPri(self):
        print("Layer 1 Weights:\n", self.layer1.synapticWeights)
        print("Layer 2 Weights:\n", self.layer2.synapticWeights)

if __name__ == '__main__':
    np.random.seed(1)

    layer1 = neuronLayer(4, 3)
    layer2 = neuronLayer(1, 4)

    neuralNet = neuralNet(layer1, layer2)

    print("Random starting synaptic weights: ")
    neuralNet.weightPri()

    tsi = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
    tso = np.array([[0, 0, 0, 1, 1, 1]]).T

    #tsi = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]])
    #tso = np.array([[0, 1, 1, 1, 1, 0, 0]]).T

    neuralNet.train(tsi, tso, 100000)

    print("New synaptic weights after training: ")
    neuralNet.weightPri()

    print("[1, 0, 1] -> 1: ", neuralNet.test(np.array([1, 0, 1])))
