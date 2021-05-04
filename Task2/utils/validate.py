from utils.generateVocabulary import loadVocabulary
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#from utils.metrics import BLEU, CIDEr, BERT, SPICE, ROUGE, METEOR
#from utils.metrics import BLEU, CIDEr, SPICE, ROUGE, METEOR

def plotImagesAndCaptions(model, modelParam, config, dataLoader):
    is_train = False
    # dataDict = next(iter(dataLoader.myDataDicts['val']))

    fig, ax = plt.subplots()
    

    
    
    # for dataDict in dataLoader.myDataDicts['val']:
    
    dataDict = next(iter(dataLoader.myDataDicts['val']))

    for key in ['xTokens', 'yTokens', 'yWeights', 'cnn_features']:
        dataDict[key] = dataDict[key].to(model.device)
    for idx in range(dataDict['numbOfTruncatedSequences']):
        # for iter in range(1):
        xTokens = dataDict['xTokens'][:, :, idx]
        yTokens = dataDict['yTokens'][:, :, idx]
        yWeights = dataDict['yWeights'][:, :, idx]
        cnn_features = dataDict['cnn_features']
        if idx == 0:
            logits, current_hidden_state = model.net(cnn_features, xTokens, is_train)
            predicted_tokens = logits.argmax(dim=2).detach().cpu()
        else:
            logits, current_hidden_state = model.net(cnn_features, xTokens, is_train, current_hidden_state)
            predicted_tokens = torch.cat((predicted_tokens, logits.argmax(dim=2).detach().cpu()), dim=1)


    vocabularyDict = loadVocabulary(modelParam['data_dir'])
    TokenToWord = vocabularyDict['TokenToWord']

    #wordToToken
    #TokenToWord

    print('predicted_tokens.shape',predicted_tokens.shape)

    batchInd = 0

    sentence = []
    foundEnd = False
    for kk in range(predicted_tokens.shape[1]):
        word = TokenToWord[predicted_tokens[batchInd, kk].item()]
        if word == 'eeee':
            foundEnd = True
        if foundEnd == False:
            sentence.append(word)

    #print captions
    print('\n')
    print('Generated caption')
    print(" ".join(sentence))
    print('\n')
    print('Original captions:')
    for kk in range(len(dataDict['orig_captions'][batchInd])):
        print(dataDict['orig_captions'][batchInd][kk])

    print('\n')

    # show image
    imgpath = modelParam['data_dir']+modelParam['modeSetups'][0][0]+ '2017/'+dataDict['imgPaths'][batchInd]
    img = mpimg.imread(imgpath)
    plt.ion()
    ax.imshow(img)
    plt.show()
    aa = 1
    return

########################################################################################################################



