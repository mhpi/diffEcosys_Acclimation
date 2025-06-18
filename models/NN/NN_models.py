import torch
import torch.nn as nn

#####################################################################################################################
def embszs_define(cat_szs):#(cols, df):
    # Description: Define the embedding size used to convert categorical inputs into quantitative values
    # Inputs:
    # catszs: number of categories for the categorical variables (i.e. PFT)

    # Outputs:
    # emb_szs: The embedding size as a tuple
    emb_szs = [(size, min(50, (size+1)//2)) for size in cat_szs]
    return emb_szs
#####################################################################################################################
class Network_relu(nn.Module):
    
    # A feedforward neural network with ReLU activations and dropout regularization.
    
    # Architecture:
    # - Input layer taking continuous features.
    # - One or more hidden layers with ReLU activations and dropout.
    # - Output layer with ReLU activation
    
    # Parameters:
    # -----------
    # n_cont : Number of continuous input features.
    # out_sz : Number of output units
    # layers : List specifying the number of neurons in each hidden layer.
    # p      : Dropout probability applied after each hidden layer (default is 0.4).
    
    def __init__(self, n_cont, out_sz, layers, p = 0.4):
        super().__init__()
        self.emb_drop = nn.Dropout(p)
        layerlist = []
        n_in =  n_cont
        for i in layers:
            layerlist.append(nn.Linear(n_in, i))
            layerlist.append(nn.ReLU(inplace=True))
            layerlist.append(nn.Dropout(p))
            n_in = i
        layerlist.append(nn.Linear(layers[-1], out_sz))
        layerlist.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layerlist)

    def forward(self, x_cont):
        x = torch.cat([x_cont],1)
        x = self.layers(x)
        return x
#####################################################################################################################
class Network_relu_sig(nn.Module):

    # A fully connected feedforward neural network with ReLU activations and dropout regularization.
    
    # Architecture:
    # - Input layer taking continuous features.
    # - One or more hidden layers with ReLU activations and dropout.
    # - Output layer with Sigmoid activation
    
    # Parameters:
    # -----------
    # n_cont : Number of continuous input features.
    # out_sz : Number of output units. Typically 1 for binary classification.
    # layers : List of integers specifying the number of neurons in each hidden layer.
    # p      : Dropout probability applied after each hidden layer (default is 0.4).

    def __init__(self, n_cont, out_sz, layers, p = 0.4):
        super().__init__()

        self.emb_drop = nn.Dropout(p)
        layerlist = []
        n_in =  n_cont
        for i in layers:
            layerlist.append(nn.Linear(n_in, i))
            layerlist.append(nn.ReLU(inplace = True))
            layerlist.append(nn.Dropout(p))
            n_in = i
        layerlist.append(nn.Linear(layers[-1], out_sz))
        layerlist.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layerlist)

    def forward(self, x_cont):

        x = torch.cat([x_cont],1)
        x = self.layers(x)
        return x
#####################################################################################################################
class TabularModel_sig(nn.Module):
       
    # A flexible neural network for tabular data that handles both categorical and continuous inputs.
    
    # Architecture:
    # - Embedding layers for categorical inputs.
    # - Concatenation of embedded categorical and continuous inputs.
    # - One or more hidden layers with Sigmoid activations and dropout.
    # - Output layer with Sigmoid activation
    
    # Parameters:
    # -----------
    # emb_szs : List of (num_categories, embedding_dim) for each categorical variable.
    # n_cont  : Number of continuous input features.
    # out_sz  : umber of output units
    # layers  : List specifying the number of units in each hidden layer.
    # p       : Dropout probability applied after each hidden layer (default is 0.4).
    
    def __init__(self, emb_szs, n_cont, out_sz, layers, p = 0.4):
        super().__init__()

        self.embeds = nn.ModuleList([nn.Embedding(ni,nf) for ni,nf in emb_szs])
        self.emb_drop = nn.Dropout(p)
        layerlist = []
        n_embs = sum([nf for ni,nf in emb_szs])
        n_in = n_embs + n_cont
        for i in layers:
            layerlist.append(nn.Linear(n_in, i))
            layerlist.append(nn.Sigmoid())
            layerlist.append(nn.Dropout(p))
            n_in = i
        layerlist.append(nn.Linear(layers[-1], out_sz))
        layerlist.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layerlist)

    def forward(self, x_cat, x_cont):
        embeddings = []
        for i,e in enumerate(self.embeds):
            embeddings.append(e(x_cat[:,i]))

        x = torch.cat(embeddings,1)
        x = self.emb_drop(x)
        #x_cont = self.bn_cont(x_cont)
        x = torch.cat([x,x_cont],1)
        x = self.layers(x)
        return x
#####################################################################################################################