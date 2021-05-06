from torch import optim

from cocoSource_xcnnfused import loss_fn

class Model():
    def __init__(self, config, modelParam, imageCaptionModel):
        self.start_epoch = 0

        if modelParam['cuda']['use_cuda']:
            self.device = f"cuda:{modelParam['cuda']['device_idx']}"
        else:
            self.device = "cpu"

        self.config         = config
        self.modelParam     = modelParam

        self.net     = imageCaptionModel(config)
        self.net.to(self.device)
        self.loss_fn = loss_fn

        if config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(self.net.parameters(), lr=config['learningRate']['lr'], weight_decay=config['weight_decay'])
        elif config['optimizer'] == 'adamW':
            self.optimizer = optim.AdamW(self.net.parameters(), lr=config['learningRate']['lr'], weight_decay=config['weight_decay'])
        elif config['optimizer'] == 'SGD':
            self.optimizer = optim.SGD(self.net.parameters(), lr=config['learningRate']['lr'], weight_decay=config['weight_decay'])
        elif config['optimizer'] == 'RMSprop':
            self.optimizer = optim.RMSprop(self.net.parameters(), lr=config['learningRate']['lr'], weight_decay=config['weight_decay'])
        else:
            raise Exception('invalid optimizer')
            
        self.scheduler = None    
        if (config['scheduler_milestones'] is not None) and (config['scheduler_factor'] is not None) :
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=config['scheduler_milestones'], gamma= config['scheduler_factor'])

            
            
        return
