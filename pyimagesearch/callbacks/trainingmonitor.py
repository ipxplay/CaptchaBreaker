from keras.callbacks import BaseLogger
import matplotlib.pyplot as plt
import numpy as np
import json
import os


class TrainingMonitor(BaseLogger):
    def __init__(self, figPath, jsonPath=None, startAt=0):
        # store the output path for the figure, the path to the json
        # serialized file, and the starting epoch
        super(TrainingMonitor, self).__init__()
        self.figPath = figPath
        self.jsonPath = jsonPath
        self.startAt = startAt

    def on_train_begin(self, logs=None):
        # initialize the history dictionary
        self.H = {}

        if self.jsonPath is not None:
            if os.path.exists(self.jsonPath):
                self.H = json.loads(open(self.jsonPath).read())

                if self.startAt > 0:
                    for k in self.H.keys():
                        self.H[k] = self.H[k][:self.startAt]

    def on_epoch_end(self, epoch, logs=None):
        for (k, v) in logs.items():
            l = self.H.get(k, [])
            l.append(v)
            self.H[k] = l

        if self.jsonPath is not None:
            with open(self.jsonPath, 'w') as f:
                f.write(json.dumps(self.H))

        if len(self.H['loss']) > 1:
            N = np.arange(1, len(self.H['loss']) + 1)
            plt.style.use('ggplot')
            plt.figure()
            plt.plot(N, self.H['loss'], label='train_loss')
            plt.plot(N, self.H['val_loss'], label='val_loss')
            # plt.plot(N, self.H['acc'], label='train_acc')
            # plt.plot(N, self.H['val_acc'], label='val_acc')
            plt.title(
                f"Training Loss [Epoch {len(self.H['loss'])}]")
            plt.xlabel('Epoch #')
            plt.ylabel('Loss')
            plt.legend()

            plt.savefig(self.figPath)
            plt.close()
