import os

import scipy
import tensorflow as tf
import matplotlib.pyplot as plt


class LossHistory(tf.keras.callbacks.Callback):

    def __init__(self, log_dir):
        import datetime
        curr_time = datetime.datetime.now()
        time_str = datetime.datetime.strftime(curr_time, '%Y-%m-%d_%H-%M-%S')
        self.log_dir = log_dir
        self.time_str = time_str
        self.save_path = os.path.join(self.log_dir, 'loss_' + time_str)
        self.losses = []
        self.val_losses = []
        os.makedirs(self.save_path)

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('loss'))
        self.losses.append(logs.get('val_loss'))
        with open(os.path.join(self.save_path, 'epoch_loss_' + self.time_str + ".txt"), 'w',encoding='utf-8') as f:
            f.write(str(logs.get('loss')) + '\n')

        with open(os.path.join(self.save_path, 'epoch_val_loss_' + self.time_str + ".txt"), 'w', encoding='utf-8') as f:
            f.write(str(logs.get('val_loss')) + '\n')

        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'r', linewidth=2, label='loss')
        plt.plot(iters, self.val_losses, 'b', linewidth=2, label='val_loss')
        # TODO 绘制这个有何作用
        # try:
        #     if len(self.losses) < 25:
        #         num = 5
        #     else:
        #         num = 15
        #
        #     plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle='--', linewidth=2,
        #              label='smooth train loss')
        #     plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle='--', linewidth=2,
        #              label='smooth val loss')
        # except:
        #     pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('A Loss Curve')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.save_path, "epoch_loss_" + str(self.time_str) + ".png"))

        plt.cla()
        plt.close("all")

