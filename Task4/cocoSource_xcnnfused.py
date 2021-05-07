from torch import nn
import torch.nn.functional as F
import torch
import numpy as np


######################################################################################################################
class imageCaptionModel(nn.Module):
    def __init__(self, config):
        super(imageCaptionModel, self).__init__()
        """
        "imageCaptionModel" is the main module class for the image captioning network

        Args:
            config: Dictionary holding neural network configuration

        Returns:
            self.Embedding  : An instance of nn.Embedding, shape[vocabulary_size, embedding_size]
            self.inputlayer : An instance of nn.Linear, shape[number_of_cnn_features, hidden_state_sizes]
            self.rnn        : An instance of RNN
            self.outputlayer: An instance of nn.Linear, shape[hidden_state_sizes, vocabulary_size]
        """
        self.config = config
        self.vocabulary_size = config['vocabulary_size']
        self.embedding_size = config['embedding_size']
        self.number_of_cnn_features = config['number_of_cnn_features']
        self.hidden_state_sizes = config['hidden_state_sizes']
        self.num_rnn_layers = config['num_rnn_layers']
        self.cell_type = config['cellType']
        self.last_layer_size = 10

        self.Embedding = nn.Embedding(self.vocabulary_size, self.embedding_size)

        self.outputlayer = nn.Linear(self.hidden_state_sizes, self.vocabulary_size)
        self.nnmapsize = 512  # the output size for the image features after the processing via self.inputLayer

        self.inputlayer = nn.Sequential(
            nn.MaxPool1d(1),
            nn.Dropout(p=0.25),
            nn.Conv1d(self.number_of_cnn_features, self.nnmapsize, kernel_size=1),
            nn.BatchNorm1d(self.nnmapsize),
            nn.LeakyReLU()
        )

        self.attentionlayer = nn.Sequential(
            nn.Dropout(p=0.25),
            nn.Linear(2*self.hidden_state_sizes, 50),
            nn.LeakyReLU(),
            nn.Linear(50, 10),
            nn.Softmax()
        )

        self.simplifiedrnn = False

        if True == self.simplifiedrnn:
            if self.cell_type != 'RNN':
                print('unsupported combi: True == self.simplifiedrnn and self.cell_type other than RNN', self.cell_type)
                exit()

            if self.config['num_rnn_layers'] != 1:
                print('unsupported combi: True == self.simplifiedrnn and self.config[num_rnn_layers] !=1',
                      self.config['num_rnn_layers'])
                exit()

            self.rnn = RNN_onelayer_simplified(input_size=self.embedding_size + self.nnmapsize,
                                               hidden_state_size=self.hidden_state_sizes)

        else:
            self.rnn = RNN(input_size=self.embedding_size + self.nnmapsize, hidden_state_size=self.hidden_state_sizes,
                           num_rnn_layers=self.num_rnn_layers, last_layer_size=self.last_layer_size, cell_type=self.cell_type)

        return

    def forward(self, cnn_features, xTokens, is_train, current_hidden_state=None):
        """
        Args:
            cnn_features        : Features from the CNN network, shape[batch_size, number_of_cnn_features]
            xTokens             : Shape[batch_size, truncated_backprop_length]
            is_train            : "is_train" is a flag used to select whether or not to use estimated token as input
            current_hidden_state: If not None, "current_hidden_state" should be passed into the rnn module
                                  shape[num_rnn_layers, batch_size, hidden_state_sizes]

        Returns:
            logits              : Shape[batch_size, truncated_backprop_length, vocabulary_size]
            current_hidden_state: shape[num_rnn_layers, batch_size, hidden_state_sizes]
        """
        # ToDO
        # Get "initial_hidden_state" shape[num_rnn_layers, batch_size, hidden_state_sizes].
        # Remember that each rnn cell needs its own initial state.

        print("cnn shape: ", cnn_features.shape)


        imgfeat_processed = self.inputlayer(cnn_features.transpose(1,2))
        print(imgfeat_processed.shape)


        if current_hidden_state is None:
            if self.cell_type == 'LSTM':
                initial_hidden_state = torch.zeros((self.num_rnn_layers, cnn_features.shape[0], 2*self.hidden_state_sizes),
                                               device=torch.device('cuda'))
            else:

                initial_hidden_state = torch.zeros((self.num_rnn_layers, cnn_features.shape[0], self.hidden_state_sizes),
                                               device=torch.device('cuda'))
        else:
            initial_hidden_state = current_hidden_state


        # use self.rnn to calculate "logits" and "current_hidden_state"
        logits, current_hidden_state_out = self.rnn(xTokens, imgfeat_processed, initial_hidden_state, self.outputlayer,
                                                    self.Embedding, is_train)

        return logits, current_hidden_state_out


######################################################################################################################


class RNN_onelayer_simplified(nn.Module):
    def __init__(self, input_size, hidden_state_size):
        super(RNN_onelayer_simplified, self).__init__()

        self.input_size = input_size
        self.hidden_state_size = hidden_state_size

        self.cells = nn.ModuleList(
            [RNNsimpleCell(hidden_state_size=self.hidden_state_size, input_size=self.input_size)])

    def forward(self, xTokens, baseimgfeat, initial_hidden_state, outputlayer, Embedding, is_train=True):

        if is_train == True:
            seqLen = xTokens.shape[1]  # truncated_backprop_length
        else:
            seqLen = 40  # Max sequence length to be generated

        # get input embedding vectors for the whole sequence
        embed_input_vec = Embedding(input=xTokens)  # (batch, seq, feature = 300)

        # first input token, that why indexing by [:,0,:]
        tokens_vector = embed_input_vec[:, 0, :]  # (batch,  feature )

        # Use for loops to run over "seqLen" and "self.num_rnn_layers" to calculate logits
        logits_series = []

        current_state = initial_hidden_state
        for kk in range(seqLen):
            updatedstate = torch.zeros_like(current_state)

            # this is for a one-layer RNN
            # in a 2 layer rnn you have to iterate here through the 2 layers
            # and input at each layer the correct input ,
            # the input at higher layers will be the hidden state from the layer below
            lvl0input = torch.cat((baseimgfeat, tokens_vector), dim=1)  # what
            # note that      current_state has 3 dims ( ...len(current_state.shape)==3... ) with first dimension having only 1 element, while the rnn cell needs a state with 2 dims as input

            current_state = torch.squeeze(current_state, dim=0)
            updatedstate[0, :] = self.cells[0](lvl0input, current_state)
            # RNN cell is used here #uses lvl0input and the hiddenstate

            # for a 2 layer rnn you do this for every kk, but you do this when you are *at the last layer of the rnn* for the current sequence index kk
            # apply the output layer to the updated state
            logitskk = outputlayer(
                updatedstate[0, :])  # note: for LSTM you use only the part which corresponds to the hidden state
            # find the next predicted output element
            tokens = torch.argmax(logitskk, dim=1)
            logits_series.append(logitskk)

            # update this at after consuming every sequence element
            current_state = updatedstate
            # set what will be the next input token
            # training:  the next vector from embed_input_vec which comes from the input sequence
            # prediction: the last predicted token
            if kk < seqLen - 1:
                if is_train == True:
                    tokens_vector = embed_input_vec[:, kk + 1, :]
                elif is_train == False:
                    tokens_vector = Embedding(tokens)

        # Produce outputs
        logits = torch.stack(logits_series, dim=1)

        return logits, current_state


class RNN(nn.Module):
    def __init__(self, input_size, hidden_state_size, num_rnn_layers, last_layer_size, cell_type='GRU'):
        super(RNN, self).__init__()
        """
        Args:
            input_size (Int)        : embedding_size
            hidden_state_size (Int) : Number of features in the rnn cells (will be equal for all rnn layers)
            num_rnn_layers (Int)    : Number of stacked rnns
            cell_type               : Whether to use vanilla or GRU cells

        Returns:
            self.cells              : A nn.ModuleList with entities of "RNNCell" or "GRUCell"
        """
        self.input_size = input_size
        self.hidden_state_size = hidden_state_size
        self.num_rnn_layers = num_rnn_layers
        self.cell_type = cell_type
        self.last_layer_size = last_layer_size

        # TODO
        input_size_list = [input_size, last_layer_size]

        # input_size_list should have a length equal to the number of layers and input_size_list[i] should contain the input size for layer i

        # TODO
        # Your task is to create a list (self.cells) of type "nn.ModuleList" and populated it with cells of type "self.cell_type" - depending on the number of rnn layers

        #if cell_type == 'GRU':
        #    self.cells = nn.ModuleList([GRUCell(hidden_state_size=self.hidden_state_size, input_size=input_size) for i in range(self.num_rnn_layers)])
        #elif cell_type == 'LSTM':
        self.cells = nn.ModuleList([LSTMCell(hidden_state_size=self.hidden_state_size, input_size=input_size) for i in range(self.num_rnn_layers)])

        return

    def forward(self, xTokens, baseimgfeat, initial_hidden_state, outputLayer, Embedding, is_train=True):
        """
        Args:
            xTokens:        shape [batch_size, truncated_backprop_length]
            initial_hidden_state:  shape [num_rnn_layers, batch_size, hidden_state_size]
            outputLayer:    handle to the last fully connected layer (an instance of nn.Linear)
            Embedding:      An instance of nn.Embedding. This is the embedding matrix.
            is_train:       flag: whether or not to feed in the predicated token vector as input for next step

        Returns:
            logits        : The predicted logits. shape[batch_size, truncated_backprop_length, vocabulary_size]
            current_state : The hidden state from the last iteration (in time/words).
                            Shape[num_rnn_layers, batch_size, hidden_state_sizes]
        """
        if is_train == True:
            seqLen = xTokens.shape[1]  # truncated_backprop_length
        else:
            seqLen = 40  # Max sequence length to be generated

        # While iterate through the (stacked) rnn, it may be easier to use lists instead of indexing the tensors.
        # You can use "list(torch.unbind())" and "torch.stack()" to convert from pytorch tensor to lists and back again.

        # get input embedding vectors
        embed_input_vec = Embedding(input=xTokens)  # .clone())    #(batch, seq, feature = 300)
        # print(embed_input_vec.shape)
        # exit()
        tokens_vector = embed_input_vec[:, 0, :]  # dim: (batch,  feature ) # the first input sequence element

        # Use for loops to run over "seqLen" and "self.num_rnn_layers" to calculate logits
        logits_series = []

        # current_state = list(torch.unbind(initial_hidden_state, dim=0))
        current_state = initial_hidden_state
        for kk in range(seqLen):
            updatedstate = torch.zeros_like(current_state)

            # TODO
            # you need to:
            # create your lvl0input,
            # update the hidden cell state for every layer with inputs depending on the layer index
            # if you are at the last layer, then produce logitskk, tokens , run a logits_series.append(logitskk),
            # see the simplified rnn for the one layer version

            print("baseimgfeat: ", baseimgfeat.shape)

            lvl0input = torch.cat((baseimgfeat, tokens_vector), dim=1)
            #print("Current shape: ", current_state.shape)
            #updatedstate[0, :] = self.cells[0](lvl0input, current_state[0, :, :])

            for layer in range(self.num_rnn_layers):
                if layer == 0:
                    updatedstate[layer, :] = self.cells[layer].forward(lvl0input, current_state[layer-1,:])
                if layer > 0:
                    attention = attentionlayer(current_state[layer-1,:])
                    updatedstate[layer, :] = self.cells[layer].forward(lvl0input, attention)


            out = updatedstate[self.num_rnn_layers - 1, : , :self.hidden_state_size]

            #print("out: ", out.shape)

            logitskk = outputLayer(out)


            tokens = torch.argmax(logitskk, dim=1)
            logits_series.append(logitskk)

            current_state = updatedstate


            if kk < seqLen - 1:
                if is_train == True:
                    tokens_vector = embed_input_vec[:, kk + 1, :]
                elif is_train == False:
                    tokens_vector = Embedding(tokens)

        # Produce outputs
        logits = torch.stack(logits_series, dim=1)
        # current_state = torch.stack(current_state, dim=0)
        return logits, current_state


########################################################################################################################
class GRUCell(nn.Module):
    def __init__(self, hidden_state_size, input_size):
        super(GRUCell, self).__init__()
        """
        Args:
            hidden_state_size: Integer defining the size of the hidden state of rnn cell
            inputSize: Integer defining the number of input features to the rnn

        Returns:
            self.weight_u: A nn.Parametere with shape [hidden_state_sizes+inputSize, hidden_state_sizes]. Initialized using
                           variance scaling with zero mean.

            self.weight_r: A nn.Parametere with shape [hidden_state_sizes+inputSize, hidden_state_sizes]. Initialized using
                           variance scaling with zero mean.

            self.weight: A nn.Parametere with shape [hidden_state_sizes+inputSize, hidden_state_sizes]. Initialized using
                         variance scaling with zero mean.

            self.bias_u: A nn.Parameter with shape [1, hidden_state_sizes]. Initialized to zero.

            self.bias_r: A nn.Parameter with shape [1, hidden_state_sizes]. Initialized to zero.

            self.bias: A nn.Parameter with shape [1, hidden_state_sizes]. Initialized to zero.

        Tips:
            Variance scaling:  Var[W] = 1/n
        """

        #TODO REMOVE
        """
        print("Input size:   ", input_size)
        print("hidden size:  ", hidden_state_size)
        print("input+weigh:  ", input_size + hidden_state_size)
        """
        self.input_size = input_size
        self.hidden_state_sizes = hidden_state_size

        self.weight_u = nn.Parameter(
            torch.randn(input_size + hidden_state_size, hidden_state_size) / np.sqrt(input_size + hidden_state_size))
        self.bias_u = nn.Parameter(torch.zeros(1, hidden_state_size))

        self.weight_r = nn.Parameter(
            torch.randn(input_size + hidden_state_size, hidden_state_size) / np.sqrt(input_size + hidden_state_size))
        self.bias_r = nn.Parameter(torch.zeros(1, hidden_state_size))

        self.weight = nn.Parameter(
            torch.randn(input_size + hidden_state_size, hidden_state_size) / np.sqrt(input_size + hidden_state_size))
        self.bias = nn.Parameter(torch.zeros(1, hidden_state_size))
        return

    def forward(self, x, state_old):
        ""
        """
        Args:
            x: tensor with shape [batch_size, inputSize]
            state_old: tensor with shape [batch_size, hidden_state_sizes]

        Returns:
            state_new: The updated hidden state of the recurrent cell. Shape [batch_size, hidden_state_sizes]

        """

        #print("Weight_r: ", self.weight_r.shape)
        #print("x: ", x.shape)
        #print("state_old: ", state_old.shape)
        #print("bias_r: ", self.bias_r.shape)

        input_cat = torch.cat((x, state_old), dim=1)
        reset = torch.mm(input_cat, self.weight_r) + self.bias_r
        reset = torch.sigmoid(reset)

        update = torch.mm(input_cat, self.weight_u) + self.bias_u
        update = torch.sigmoid(update)

        #TODO REMOVE
        """
        print("reset:     ", reset.shape)
        print("state_old: ", state_old.shape)
        print("x:         ", x.shape)
        """

        product = torch.mul(reset, state_old)

        #TODO REMOVE
        #print("product:   ", product.shape)

        #reset_cat = torch.cat((x, product), dim=1)

        #TODO REMOVE
        """
        print("reset_cat: ", reset_cat.shape)
        print("weight:    ", self.weight.shape)
        """

        cand_hidden = torch.mm(x, self.weight_r[:self.input_size,:]) + torch.mm(product, self.weight[self.hidden_state_size:,:]) + self.bias
        cand_hidden = torch.tanh(cand_hidden)

        #TODO REMOVE
        #print("cand_hidde: ", cand_hidden.shape)

        hidden_state_update = torch.mul(update, state_old) + torch.mul(torch.sub(1,update), cand_hidden)
        state_new = hidden_state_update
        return state_new


######################################################################################################################
class RNNsimpleCell(nn.Module):
    def __init__(self, hidden_state_size, input_size):
        super(RNNsimpleCell, self).__init__()
        """
        Args:
            hidden_state_size: Integer defining the size of the hidden state of rnn cell
            inputSize: Integer defining the number of input features to the rnn

        Returns:
            self.weight: A nn.Parameter with shape [hidden_state_sizes+inputSize, hidden_state_sizes]. Initialized using
                         variance scaling with zero mean.

            self.bias: A nn.Parameter with shape [1, hidden_state_sizes]. Initialized to zero.

        Tips:
            Variance scaling:  Var[W] = 1/n
        """
        self.hidden_state_size = hidden_state_size

        self.weight = nn.Parameter(
            torch.randn(input_size + hidden_state_size, hidden_state_size) / np.sqrt(input_size + hidden_state_size))
        self.bias = nn.Parameter(torch.zeros(1, hidden_state_size))
        return

    def forward(self, x, state_old):
        """
        Args:
            x: tensor with shape [batch_size, inputSize]
            state_old: tensor with shape [batch_size, hidden_state_sizes]

        Returns:
            state_new: The updated hidden state of the recurrent cell. Shape [batch_size, hidden_state_sizes]

        """
        x2 = torch.cat((x, state_old), dim=1)
        state_new = torch.tanh(torch.mm(x2, self.weight) + self.bias)
        return state_new


######################################################################################################################

class LSTMCell(nn.Module):
    def __init__(self, hidden_state_size, input_size):
        super(LSTMCell, self).__init__()
        """
        Args:
            hidden_state_size: Integer defining the size of the hidden state of rnn cell
            inputSize: Integer defining the number of input features to the rnn

            note: the actual tensor has 2*hidden_state_size because it contains hiddenstate and memory cell
        Returns:
            self.weight_f ...

        Tips:
            Variance scaling:  Var[W] = 1/n
        """
        self.hidden_state_size = hidden_state_size

        # TODO:
        self.weight_f = nn.Parameter(
            torch.randn(input_size + 2*hidden_state_size, hidden_state_size) / np.sqrt(input_size + 2*hidden_state_size))
        self.bias_f = nn.Parameter(torch.zeros(1, hidden_state_size))

        self.weight_i = nn.Parameter(
            torch.randn(input_size + 2*hidden_state_size, hidden_state_size) / np.sqrt(input_size + 2*hidden_state_size))
        self.bias_i = nn.Parameter(torch.zeros(1, hidden_state_size))

        self.weight_meminput = nn.Parameter(
            torch.randn(input_size + 2*hidden_state_size, hidden_state_size) / np.sqrt(input_size + 2*hidden_state_size))
        self.bias_meminput = nn.Parameter(torch.zeros(1, hidden_state_size))

        self.weight_o = nn.Parameter(
            torch.randn(input_size + 2*hidden_state_size, hidden_state_size) / np.sqrt(input_size + 2*hidden_state_size))
        self.bias_o = nn.Parameter(torch.zeros(1, hidden_state_size))

        return

    def forward(self, x, state_old):
        """
        Args:
            x: tensor with shape [batch_size, inputSize]
            state_old: tensor with shape [batch_size, 2*hidden_state_sizes]

        Returns:
            state_new: The updated hidden state of the recurrent cell. Shape [batch_size, hidden_state_sizes]

        """


        #print("x:         ", x.shape)
        #print("state_old: ", state_old.shape)
        #print()

        # TODO:
        input_cat = torch.cat((x, state_old), dim=1)

        #print("input_cat: ", input_cat.shape)
        #print("weight_i : ", self.weight_i.shape)
        #print("bias_i:    ", self.bias_i.shape)

        input_gate = torch.sigmoid(torch.mm(input_cat, self.weight_i) + self.bias_i)
        #input_gate = torch.sigmoid(input_gate)

        forget_gate = torch.sigmoid(torch.mm(input_cat, self.weight_f) + self.bias_f)
        #forget_gate = torch.sigmoid(forget_gate)

        output_gate = torch.sigmoid(torch.mm(input_cat, self.weight_o) + self.bias_o)
        #output_gate = torch.sigmoid(output_gate)

        candidate_memory = torch.mm(input_cat, self.weight_meminput) + self.bias_meminput
        candidate_mem_tanh = torch.tanh(candidate_memory.clone())

        #print("forget:    ", forget_gate.shape)
        #print("state_old: ", state_old.shape)
        #print("state_old_orig: ", state_old.shape)

        memory_cell = torch.mul(forget_gate, state_old[:,self.hidden_state_size:]) + torch.mul(input_gate, candidate_mem_tanh)

        memory_cell_tanh = torch.tanh(memory_cell.clone())

        #print("Memory shape: ", memory_cell_tanh.shape)

        state_new = torch.cat((torch.mul(output_gate, torch.tanh(memory_cell_tanh)), memory_cell), dim=1)

        #state_new = torch.cat((hidden_state_update, memory_cell_tanh), dim=1)

        #print("state_new:  ", state_new.shape)
        #print("")

        return state_new


######################################################################################################################
def loss_fn(logits, yTokens, yWeights):
    """
    Weighted softmax cross entropy loss.

    Args:
        logits          : shape[batch_size, truncated_backprop_length, vocabulary_size]
        yTokens (labels): Shape[batch_size, truncated_backprop_length]
        yWeights        : Shape[batch_size, truncated_backprop_length]. Add contribution to the total loss only from words exsisting
                          (the sequence lengths may not add up to #*truncated_backprop_length)

    Returns:
        sumLoss: The total cross entropy loss for all words
        meanLoss: The averaged cross entropy loss for all words

    Tips:
        F.cross_entropy
    """
    eps = 0.0000000001  # used to not divide on zero

    logits = logits.view(-1, logits.shape[2])
    yTokens = yTokens.view(-1)
    yWeights = yWeights.view(-1)
    losses = F.cross_entropy(input=logits, target=yTokens, reduction='none')

    sumLoss = (losses * yWeights).sum()
    meanLoss = sumLoss / (yWeights.sum() + eps)

    return sumLoss, meanLoss

# ########################################################################################################################
# if __name__ == '__main__':
#
#     lossDict = {'logits': logits,
#                 'yTokens': yTokens,
#                 'yWeights': yWeights,
#                 'sumLoss': sumLoss,
#                 'meanLoss': meanLoss
#     }
#
#     sumLoss, meanLoss = loss_fn(logits, yTokens, yWeights)
#
