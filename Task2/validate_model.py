from utils.dataLoader import DataLoaderWrapper
from utils.saverRestorer import SaverRestorer
from utils.model import Model
from utils.trainer import Trainer
from utils.validate import plotImagesAndCaptions
from utils.validate_metrics import validateCaptions

from cocoSource_xcnnfused import imageCaptionModel # here you plug in your modelfile depending on what you have developed: simple rnn, 2 layer, or attention, if you have 3 modelfiles a.py b.py c.py then you do: from a import ... or you have one file with n different imgcapmodels

def main(config, modelParam):
    # create an instance of the model you want
    model = Model(config, modelParam, imageCaptionModel)

    # create an instacne of the saver and resoterer class
    saveRestorer = SaverRestorer(config, modelParam)

    if modelParam['inference'] == True:
        model        = saveRestorer.restore(model)

    # create your data generator
    dataLoader = DataLoaderWrapper(config, modelParam)

    # here you train your model
    if modelParam['inference'] == False:
        # create trainer and pass all the previous components to it
        trainer = Trainer(model, modelParam, config, dataLoader, saveRestorer)
        trainer.train()

    #plotImagesAndCaptions
    if modelParam['inference'] == True:
        #plotImagesAndCaptions(model, modelParam, config, dataLoader)
        validateCaptions(model, modelParam, config, dataLoader)

        plotImagesAndCaptions(model, modelParam, config, dataLoader)



    return


if __name__ == '__main__':
    data_dir = '../../../../shared/IN5400/dataforall/mandatory2/data/coco/'


    #train
    modelParam = {
        'batch_size': 128,  # Training batch size
        'cuda': {'use_cuda': True,  # Use_cuda=True: use GPU
                 'device_idx': 0},  # Select gpu index: 0,1,2,3
        'numbOfCPUThreadsUsed': 10,  # Number of cpu threads use in the dataloader
        'numbOfEpochs': 99,  # Number of epochs
        'data_dir': data_dir,  # data directory
        'img_dir': 'loss_images_test/',
        'modelsDir': 'storedModels_test/',
        'modelName': 'model_0/',  # name of your trained model
        'restoreModelLast': 0,
        'restoreModelBest': 0,
        'modeSetups': [['train', True], ['val', True]],
        'inNotebook': False,  # If running script in jupyter notebook
        'inference': True
    }

    config = {
        'optimizer': 'adamW',  # 'SGD' | 'adam' | 'RMSprop' | 'adamW'
        'learningRate': {'lr': 0.001},  # learning rate to the optimizer
        'weight_decay': 0.00001,  # weight_decay value
        'number_of_cnn_features': 2048,  # Fixed, do not change
        'embedding_size': 300,  # word embedding size
        'vocabulary_size': 10000,  # number of different words
        'truncated_backprop_length': 25,
        'hidden_state_sizes': 512,  #
        'num_rnn_layers': 2,  # number of stacked rnn's
        'scheduler_milestones': [75,90], #45,70 end at 80? or 60, 80
        'scheduler_factor': 0.2, #+0.25 dropout
        #'featurepathstub': 'detectron2vg_features' ,
        #'featurepathstub': 'detectron2m_features' ,
        #'featurepathstub': 'detectron2cocov3_tenmfeatures' ,
        'featurepathstub': 'detectron2_lim10maxfeatures' ,
        'cellType':  'LSTM' #'GRU'  # RNN or GRU or LSTM??
    }

    if modelParam['inference'] == True:
        modelParam['batch_size'] = 64
        modelParam['modeSetups'] = [['val', False]]
        modelParam['restoreModelBest'] = 1

    main(config, modelParam)

    aa = 1
