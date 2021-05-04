import matplotlib.pyplot as plt
from time import sleep

import os

class Plotter():
    def __init__(self, modelParam, config):
    
        self.bestvalmeasure=None
    
        self.modelParam = modelParam
        self.config     = config
        # plt.figure()
        self.fig, self.ax = plt.subplots(1,2)
        # if not self.modelParam['inNotebook']:
        # plt.show()
        self.fig.show()
        sleep(0.1)
        self.ax[0].set_ylabel('Loss')
        self.ax[0].set_xlabel('epoch [#]')

        # train_line = self.ax.plot([],[],color='blue', label='Train', marker='.', linestyle="")
        # val_line   = self.ax.plot([], [], color='red', label='Validation', marker='.', linestyle="")

        train_line = self.ax[0].plot([],[],color='blue', label='Train', marker='.', linestyle="")
        val_line   = self.ax[0].plot([], [], color='red', label='Validation', marker='.', linestyle="")
        self.ax[0].legend(handles=[train_line[0], val_line[0]])
        self.ax[0].set_axisbelow(True)
        self.ax[0].grid()
        self.ax[0].set_ylim([0, 5])
        self.ax[1].grid()
        self.ax[1].set_ylim([0, 0.39])
        sleep(0.1)
        return


    def update(self, current_epoch, loss, mode):
        if mode=='train':
            color = 'b'
        else:
            color = 'r'

        if self.modelParam['inNotebook']:
            self.ax[0].scatter(current_epoch, loss, c=color)
            self.ax[0].set_ylim(bottom=0, top=self.ax[0].get_ylim()[1])
            self.fig.canvas.draw()
            sleep(0.1)
        else:
            self.ax[0].scatter(current_epoch, loss, c=color)
            self.ax[0].set_ylim(bottom=0, top=self.ax[0].get_ylim()[1])
            self.fig.canvas.draw()
            sleep(0.1)
            self.save()
        return
        
    def update_withval(self, current_epoch, loss, valmeasure, mode):
        if mode=='train':
            color = 'b'
        else:
            color = 'r'

        if self.bestvalmeasure is None:
          self.bestvalmeasure = valmeasure
        elif self.bestvalmeasure < valmeasure :
          self.bestvalmeasure = valmeasure
        print('\n\n\ncurrent best val measure and current valmeasure ',self.bestvalmeasure, valmeasure)

        if self.modelParam['inNotebook']:
            self.ax[0].scatter(current_epoch, loss, c=color)
            self.ax[0].set_ylim(bottom=0, top=self.ax[0].get_ylim()[1])
            
            self.ax[1].scatter(current_epoch, valmeasure, c='r')
            self.ax[1].set_ylim(bottom=0, top=self.ax[1].get_ylim()[1])
            
            self.fig.canvas.draw()
            sleep(0.1)
        else:
            self.ax[0].scatter(current_epoch, loss, c=color)
            self.ax[0].set_ylim(bottom=0, top=self.ax[0].get_ylim()[1])
            
            self.ax[1].scatter(current_epoch, valmeasure, c='r')
            self.ax[1].set_ylim(bottom=0, top=0.5)
            
            self.fig.canvas.draw()
            sleep(0.1)
            self.save()
        return

    def save(self):
        # path = self._getPath()
        pt = './loss_images/'
        if not os.path.isdir(pt):
          os.makedirs(pt)
        path = pt+self.modelParam['modelName'][:-1]
        self.fig.savefig(path+'.png')
        return

    def _getPath(self):
        keys = self.config.keys()
        path = 'loss_images/'
        first=1
        for key in keys:
            if first!=1:
                path += '_'
            else:
                first=0
            element = self.config[key]
            if isinstance(element, str):
                path += element
            elif isinstance(element, int):
                path += key+str(element)
            elif isinstance(element, float):
                path += key+str(element)
            elif isinstance(element, list):
                path += ''
                for elm in element:
                    path += str(elm)
            elif isinstance(element, dict):
                path += ''
                for elKey, elVal in element.items():
                    path += str(elKey) + str(elVal).replace('.', '_')
            else:
                raise Exception('Unknown element in config')
        return path
