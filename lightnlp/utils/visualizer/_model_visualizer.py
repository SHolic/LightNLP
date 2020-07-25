class ModelVisualizer:
    def __init__(self):
        pass

    @staticmethod
    def visualize(model):
        for name, param in model.named_parameters():
            print(name, param.shape, param.device, param.requires_grad)

