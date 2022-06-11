from os import terminal_size
import numpy as np
import pandas as pd

class neuronLayer():
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.synapticWeights = 2 * np.random.random((inputs, outputs)) - 1

class neuralNet:
    def __init__(self, layer1, layer2, layer3):
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoidDer(self, x):
        return x * (1 - x)
    
    def reLU(self, x):
        return np.maximum(-0.01,x)

    def reLUDer(self, x):
        return np.greater(x, -0.01)

    def train(self, tsi, tso, epochs):
        for i in range(epochs):
            if((i % 10000) == 0):
                print(i)
                seed = np.random.randint(0, 100000)  
                np.random.seed(seed)  
                np.random.shuffle(tsi)
                np.random.seed(seed)  
                np.random.shuffle(tso)

            layerOut1, layerOut2, layerOut3 = self.forward(tsi)

            layerErr3 = tso - layerOut3
            layerDel3 = layerErr3 * self.sigmoidDer(layerOut3)

            layerErr2 = layerDel3.dot(self.layer3.synapticWeights.T)
            layerDel2 = layerErr2 * self.sigmoidDer(layerOut2)

            layerErr1 = layerDel2.dot(self.layer2.synapticWeights.T)
            layerDel1 = layerErr1 * self.reLUDer(layerOut1)

            layerAdj1 = tsi.T.dot(layerDel1)
            layerAdj2 = layerOut1.T.dot(layerDel2)
            layerAdj3 = layerOut2.T.dot(layerDel3)

            self.layer1.synapticWeights += layerAdj1
            self.layer2.synapticWeights += layerAdj2
            self.layer3.synapticWeights += layerAdj3

    def forward(self, inputs):
        layerOut1 = self.reLUDer(np.dot(inputs, self.layer1.synapticWeights))
        layerOut2 = self.sigmoid(np.dot(layerOut1, self.layer2.synapticWeights))
        layerOut3 = self.sigmoid(np.dot(layerOut2, self.layer3.synapticWeights))
        return layerOut1, layerOut2, layerOut3

    def test(self, inputs):
        layerOut1 = self.reLUDer(np.dot(inputs, self.layer1.synapticWeights))
        layerOut2 = self.sigmoid(np.dot(layerOut1, self.layer2.synapticWeights))
        layerOut3 = self.sigmoid(np.dot(layerOut2, self.layer3.synapticWeights))

        print(f"Output\n {layerOut3}\n")
        return layerOut3

    def weightPri(self):
        print(f"Layer 1 Weights:\n{self.layer1.synapticWeights}\nLayer 2 Weights:\n{self.layer2.synapticWeights}\nLayer 3 Weights:\n{self.layer3.synapticWeights}\n")

if __name__ == '__main__':
    np.random.seed(1678678)

    layer1 = neuronLayer(9, 15)
    layer2 = neuronLayer(15, 10)
    layer3 = neuronLayer(10, 2)

    nn = neuralNet(layer1, layer2, layer3)

    # code: if first or last number is 1 output is 1 else 0
    df = pd.read_csv('hard.csv', delimiter='|', nrows=129, usecols=[0])
    df = df.IN.str.split(',', expand=True)
    trsi = pd.DataFrame(df).to_numpy(dtype=np.intc)

    df = pd.read_csv('hard.csv', delimiter='|', nrows=129, usecols=[1])
    df = df.OUT.str.split(',', expand=True)
    trso = pd.DataFrame(df).to_numpy(dtype=np.intc)

    nn.train(trsi, trso, 100000)

    df = pd.read_csv('hard.csv', delimiter='|', skiprows=[i for i in range(1,130)], usecols=[0])
    df = df.IN.str.split(',', expand=True)
    tesi = pd.DataFrame(df).to_numpy(dtype=np.intc)

    df = pd.read_csv('hard.csv', delimiter='|', skiprows=[i for i in range(1,130)], usecols=[1])
    df = df.OUT.str.split(',', expand=True)
    teso = np.int_(pd.DataFrame(df).to_numpy(dtype=np.intc))

    test = nn.test(tesi)
    rtest = np.rint(test)

    print(f"Compare:\n{teso == rtest}")
