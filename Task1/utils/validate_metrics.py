from utils.generateVocabulary import loadVocabulary
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#from utils.metrics import BLEU, CIDEr, BERT, SPICE, ROUGE, METEOR
from utils.metrics import BLEU, METEOR , CIDEr,  ROUGE

def validateCaptions(model, modelParam, config, dataLoader):
    is_train = False



    references = {}  # references (true captions) for calculating BLEU-4 score
    hypotheses = {}  # hypotheses (predictions)

    atiter=-1
    for dataDict in dataLoader.myDataDicts['val']:
    
        #atiter=0
        #dataDict = next(iter(dataLoader.myDataDicts['val']))

        for key in ['xTokens', 'yTokens', 'yWeights', 'cnn_features']:
            dataDict[key] = dataDict[key].to(model.device)
        for idx in range(dataDict['numbOfTruncatedSequences']):
            # for iter in range(1):
            xTokens = dataDict['xTokens'][:, :, idx]
            yTokens = dataDict['yTokens'][:, :, idx]
            yWeights = dataDict['yWeights'][:, :, idx]
            cnn_features = dataDict['cnn_features']
            with torch.no_grad():
              if idx == 0:
                  logits, current_hidden_state = model.net(cnn_features, xTokens, is_train)
                  predicted_tokens = logits.argmax(dim=2).detach().cpu()
              else:
                  logits, current_hidden_state = model.net(cnn_features, xTokens, is_train, current_hidden_state.detach())
                  predicted_tokens = torch.cat((predicted_tokens, logits.argmax(dim=2).detach().cpu()), dim=1)
              

        vocabularyDict = loadVocabulary(modelParam['data_dir'])
        TokenToWord = vocabularyDict['TokenToWord']
        #wordToToken
        #TokenToWord

        #print('predicted_tokens.shape',predicted_tokens.shape)

        #batchInd = 0
        
        
        
        for batchInd in range(predicted_tokens.shape[0]):

            tokensentence =[]
            sentence = []
            foundEnd = False
            for kk in range(predicted_tokens.shape[1]):
                word = TokenToWord[predicted_tokens[batchInd, kk].item()]
                if word == 'eeee':
                    foundEnd = True
                    #break
                if foundEnd == False:
                    sentence.append(word)
                    tokensentence.append( '{:d}'.format(predicted_tokens[batchInd, kk].item()) )
                    
            #print captions
            '''
            print('\n')
            print('Generated caption')
            print(" ".join(sentence))
            print('\n')
            print('Original captions:')
            for kk in range(len(dataDict['orig_captions'][batchInd])):
                print(dataDict['orig_captions'][batchInd][kk])
            print('\n')
            '''
            
            if (atiter+0)%500==0:
              print('at iter',atiter)
            
            atiter+=1
            hypotheses[atiter]=[]
            references[atiter]=[]
                    
            #predictedsentence= ' '.join(sentence)
            predictedsentence= ' '.join(tokensentence)
            hypotheses[atiter].append({'caption':predictedsentence}) 

            for kk in range(len(dataDict['allcaptionsAsTokens'][batchInd])):
                #seq2= dataDict['allcaptions'][batchInd][kk] 
                seq2= dataDict['allcaptionsAsTokens'][batchInd][kk] 
                seq2=seq2[1:-1] # drop start and end element
                
                seq2 = [  '{:d}'.format(s)  for s in seq2 ] #tokeintegers to string
                ref = ' '.join(seq2) #concat
                #print(ref,'|',predictedsentence)
                references[atiter].append({'caption':ref})

           
    
    
    results_dict = {}
    print("Calculating Evalaution Metric Scores......\n")
    avg_bleu_dict = BLEU().calculate(hypotheses, references, tokenize= False)
    bleu4 = avg_bleu_dict['bleu_4']
    
    

    avg_meteor_dict = METEOR().calculate(hypotheses, references, tokenize= False)


    results_dict.update(avg_bleu_dict)

    results_dict.update(avg_meteor_dict)
    
    avg_cider_dict = CIDEr().calculate(hypotheses, references)
    cider = avg_cider_dict['cider']
    #avg_bert_dict = BERT().calculate(hypotheses,references)
    #bert = avg_bert_dict['bert']
    avg_rouge_dict = ROUGE().calculate(hypotheses,references)
    results_dict.update(avg_cider_dict)
    results_dict.update(avg_rouge_dict)   

    print(f'Evaluation results, BLEU-4: {bleu4}, Cider: {cider},  ROUGE: {avg_rouge_dict["rouge"]}, Meteor: {avg_meteor_dict["meteor"]}')

    return results_dict

########################################################################################################################



