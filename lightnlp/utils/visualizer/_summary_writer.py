import matplotlib.pyplot as plt


class _PlotData:
    def __init__(self):
        self.x = list()
        self.y = list()

    def add(self, x=None, y=None):
        if x is not None:
            self.x.append(x)
        if y is not None:
            self.y.append(y)

    def plot(self, title):
        x, y = self.x, self.y
        if len(y) == 0:
            return
        if len(x) == 0:
            x = [i for i in range(len(y))]
        plt.figure(0)
        plt.xlabel("n_batch")
        plt.ylabel(title)
        plt.title(title)
        plt.scatter(x=x, y=y)
        plt.show()
        plt.close(0)


class SummaryWriter:
    def __init__(self):
        self.plots = dict()

    def add_scalar(self, title=None, value=None, n_iter=None):
        if title not in self.plots.keys():
            self.plots[title] = _PlotData()
        self.plots[title].add(x=n_iter, y=value)

    def plot(self, title):
        if title in self.plots.keys():
            self.plots[title].plot(title=title)
