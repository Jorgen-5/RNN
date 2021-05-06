from torch.utils.data import DataLoader
import pickle
import glob
import numpy as np
import torch

import os

class DataLoaderWrapper():
    def __init__(self, config, modelParam):

        self.data_dir = modelParam['data_dir']
        self.data_dir_train = self.data_dir + 'Train2017_'+config['featurepathstub']
        self.data_dir_val   = self.data_dir + 'Val2017_'+config['featurepathstub']
        

        
        #print(self.data_dir_train)
        #exit()

        self.truncated_backprop_length = config['truncated_backprop_length']
        self.vocabulary_size           = config['vocabulary_size']

        self.batch_size_train = modelParam['batch_size']
        self.batch_size_val   = modelParam['batch_size']

        myDatasetTrain = Coco_dataclass_cnn_features(self.data_dir_train)
        myDatasetVal   = Coco_dataclass_cnn_features(self.data_dir_val)

        myCollate_fn = CollateClass(config, modelParam)
        self.myDataDicts = {}
        self.myDataDicts['train'] = DataLoader(myDatasetTrain, batch_size=self.batch_size_train, shuffle=True, num_workers=modelParam['numbOfCPUThreadsUsed'], collate_fn=myCollate_fn)
        self.myDataDicts['val']   = DataLoader(myDatasetVal, batch_size=self.batch_size_val, shuffle=True, num_workers=modelParam['numbOfCPUThreadsUsed'], collate_fn=myCollate_fn)
        return

#######################################################################################################################
class CollateClass:
    def __init__(self, config, modelParam):
        self.truncated_backprop_length = config['truncated_backprop_length']
        self.vocabulary_size           = config['vocabulary_size']
        if modelParam['cuda']['use_cuda']:
            self.device = f"cuda:{modelParam['cuda']['device_idx']}"
        else:
            self.device = "cpu"
        return

    def __call__(self, batch):
    
        cutoff=36
    
        outDict = {}
        if len(batch[0]['cnn_features'].shape) ==1 :
            outDict['cnn_features'] = torch.tensor(np.stack([x['cnn_features'] for x in batch], axis=0))
        else:        
            outDict['cnn_features'] = torch.tensor(np.stack([x['cnn_features'][:cutoff,:] for x in batch], axis=0))

        outDict['orig_captions']    = [x['orig_captions'] for x in batch]
        outDict['captions']         = [x['captions'] for x in batch]
        outDict['captionsAsTokens'] = [x['captionsAsTokens'] for x in batch]
        outDict['imgPaths']         = [x['imgPaths'] for x in batch]
        
        
        outDict['allcaptions']         = [x['allcaptions'] for x in batch]
        outDict['allcaptionsAsTokens']         = [x['allcaptionsAsTokens'] for x in batch]
        
        outDict = self.getCaptionMatix(outDict)
        outDict['numbOfTruncatedSequences'] = outDict['yWeights'].shape[2]
        return outDict

    def getCaptionMatix(self, outDict):
        # find the length sequence and create correspinding captionMatix

        captionsAsTokens = outDict['captionsAsTokens']

        batchSize  = len(captionsAsTokens)
        seqLengths = [len(tokens) for tokens in captionsAsTokens]
        maxSeqLen  = max(seqLengths)

        divisionCount = int(np.ceil((maxSeqLen-1)/self.truncated_backprop_length))
        maxLength     = self.truncated_backprop_length*divisionCount + 1

        captionMatix  = np.zeros((batchSize, maxLength), dtype=np.int64)
        weightMatrix  = np.zeros((batchSize, maxLength), dtype=np.float32)

        mask               = np.arange(maxLength) < np.array(seqLengths)[:, None]
        captionMatix[mask] = np.concatenate(captionsAsTokens)
        weightMatrix[mask] = 1

        #set all words with index larger then "vocabulary_size" to "UNK" unknown word -> index=2
        captionMatix[captionMatix >= self.vocabulary_size] = 2

        yTokens  = captionMatix[:,1:].reshape((batchSize, self.truncated_backprop_length, divisionCount), order='F')
        yWeights = weightMatrix[:,1:].reshape((batchSize, self.truncated_backprop_length, divisionCount), order='F')

        xTokens  = captionMatix[:,:-1].reshape((batchSize, self.truncated_backprop_length, divisionCount), order='F')
        xWeights = weightMatrix[:,:-1].reshape((batchSize, self.truncated_backprop_length, divisionCount), order='F')

        outDict['xTokens']  = torch.tensor(xTokens)
        outDict['yTokens']  = torch.tensor(yTokens)
        outDict['yWeights'] = torch.tensor(yWeights)

        return outDict

########################################################################################################################
class Coco_dataclass_cnn_features():
    def __init__(self, data_dir):
    
        if not os.path.isdir(data_dir):
          print('cannot find directory', data_dir)
          exit()
    
        self.data_dir          = data_dir
        self.pickle_files_path = glob.glob(self.data_dir+'/*')
        self.captionIter       = np.zeros(len(self.pickle_files_path), dtype=int)
        return

    def __len__(self):
        return len(self.pickle_files_path)

    def __getitem__(self, item):
        with open(self.pickle_files_path[item], "rb") as input_file:
            #print(self.pickle_files_path[item])
            dataDict = pickle.load(input_file)
            
        #print(type(dataDict['cnn_features']))
        #exit()
            
        tmpOrigCaption      = dataDict['original_captions']
        tmpCaption          = dataDict['captions']
        tmpCaptionsAsTokens = dataDict['captionsAsTokens']
        imgPaths            = dataDict['imgPath']
        cnn_features        = dataDict['cnn_features']

        captionInd = self.captionIter[item]
        outDict = {}
        if len(tmpOrigCaption) <= captionInd:
            outDict['captions']         = tmpCaption[captionInd]
            outDict['captionsAsTokens'] = tmpCaptionsAsTokens[captionInd]
            captionInd = captionInd + 1
        else:
            outDict['captions']         = tmpCaption[0]
            outDict['captionsAsTokens'] = tmpCaptionsAsTokens[0]
            captionInd = 1
        self.captionIter[item]    = captionInd
        
        outDict['allcaptions']=[]
        for k in range(len(tmpCaption)):
          outDict['allcaptions'].append(tmpCaption[k] )
        outDict['allcaptionsAsTokens']=[]
        for k in range(len(tmpCaption)):
          outDict['allcaptionsAsTokens'].append(tmpCaptionsAsTokens[k] )
          
        
        outDict['orig_captions']  = tmpOrigCaption
        outDict['imgPaths']       = imgPaths
        outDict['cnn_features']   = cnn_features
        return outDict

#######################################################################################################################
