import numpy as np
import math


class SOM(object):
    def __init__(self, size, data, alpha):
        self.t = 0
        self.nodes = dict()
        self.distance = dict()
        self.rgb = dict()
        self.alpha = alpha
        self.size = size
        self.number_of_inputs = len(data['i'].count())
        for i in range(size):
            for j in range(size):
                self.nodes[(i, j)] = np.random.rand(self.number_of_inputs) / 10
                self.distance[(i, j)] = np.array([0.0, 0.0, 0.0])
                self.rgb[(i, j)] = np.array([0.0, 0.0, 0.0])
        self.sigma_initial = size
        self.sigma = self.sigma_initial
        self.sigmav2 = 2 * (self.sigma ** 2)
        self.gamma_initial = 1
        self.gamma = self.gamma_initial


    def winner(self, datax):
        x = np.array(datax)
        winner = (0, 0)
        winner_dist = math.inf
        for i in range(self.size):
            for j in range(self.size):
                temp = 0
                for k in range(self.number_of_inputs):
                    temp += (x[k] - self.nodes[(i, j)][k]) ** 2
                self.distance[(i, j)] = np.sqrt(temp)
                if self.distance[(i, j)] < winner_dist:
                    winner_dist = self.distance[(i, j)]
                    winner = (i, j)
        return winner

    def update_rgb(self, datay, winner):
        y = np.array(datay)
        a = winner[0]
        b = winner[1]
        self.rgb[(a, b)] += y
        a=y

    def update_w(self, winner, datax):
        a = winner[0]
        b = winner[1]
        for i in range(self.size):
            for j in range(self.size):
                temp_distance = ((i - a) ** 2 + (j - b) ** 2)
                if np.sqrt(temp_distance)< self.sigma:
                    delta = math.exp(- temp_distance / self.sigmav2)
                    self.nodes[(i, j)] += self.gamma * delta * (datax - self.nodes[(i, j)])


    def rgb_image(self):
        rgbArray = np.zeros((self.size,self.size,3))
        for i in range(self.size):
            for j in range(self.size):
                color = self.rgb[(i, j)]
                sum_c=sum(color)
                if sum_c > 0:
                    color = color / sum_c
                else:
                    color=[0,0,0]
                rgbArray[i][j]=np.array(color)
        return rgbArray

    def output(self, datax):
        outputList = list()
        self.winner(datax)
        for i in range(self.size):
            for j in range(self.size):
                outputList.append(self.distance[(i, j)])
        return np.array(outputList)

    def _calculate_sigma(self):
        self.sigma = self.sigma_initial * math.exp(-self.t / self.alpha)
        self.sigmav2 = 2 * (self.sigma ** 2)

    def _calculate_gamma(self):
        self.gamma = self.gamma_initial * math.exp(-self.t / self.alpha)

    def teach(self, datax, datay):
        self._calculate_gamma()
        self._calculate_sigma()
        winner = self.winner(datax)
        self.update_w(winner,datax)
        self.update_rgb(datay, winner)
        self.t += 1