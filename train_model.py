import pandas as pd
import numpy as np
import os
import pickle
from torch.serialization import load
from ts_data import TS_Data
import torch
from torch.utils.data import Dataset, DataLoader
from CNN import CNN
from metrics import nll 
import argparse 

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
#data parameters

parser.add_argument('--obs_seq_len', type=int, default=8)
parser.add_argument('--pred_seq_len', type=int, default=12)
parser.add_argument('--data_dir', default="/home/warren/Documents/UGRA/AIM/csv_files")  
#training parameters
parser.add_argument('--lr', type=float, default=0.01,help='learning rate')
parser.add_argument('--model_type', default= 'CNN', help = 'CNN, LSTM, GCNN' )
parser.add_argument('--load_model',  action = 'store_true')
parser.add_argument('--tag', default='tag',help='personal tag for the model ')
parser.add_argument('--num_epochs', type=int, default=250, help='number of epochs')  
parser.add_argument('--single_set', action = 'store_true')
parser.add_argument('--clip_grad', type=float, default=None,
                    help='gadient clipping') 
args = parser.parse_args()

obs_seq_len = args.obs_seq_len
pred_seq_len = args.pred_seq_len
def get_dataset(fdir, in_len,pred_len):
    l = os.listdir(fdir)
    datasets = [None]*len(l)
    for i in range(len(l)):
        temp = os.path.join(fdir,l[i])
        datasets[i] = TS_Data(temp, in_len,pred_len)
    dataset =  torch.utils.data.ConcatDataset(datasets)
    return dataset
fdir = args.data_dir
train_dataloader = None
test_dataloader = None
if args.single_set:
    full_dataset = get_dataset(fdir,obs_seq_len, pred_seq_len)
    train_len = int(.8* len(full_dataset))
    test_len = len(full_dataset)-train_len
    train_data, test_data = torch.utils.data.random_split(full_dataset, [train_len, test_len])
    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64)
else:
    data_dir = args.data_dir
    train_set = get_dataset(os.path.join(data_dir, "train"), obs_seq_len,pred_seq_len)
    test_set = get_dataset(os.path.join(data_dir, "valid"), obs_seq_len, pred_seq_len)
    train_dataloader = DataLoader(train_set, batch_size=64, shuffle=True )
    test_dataloader = DataLoader(test_set, batch_size=64, shuffle=False)
checkpoint_dir =  os.path.join(os.getcwd(), './checkpoint/'+args.tag+'/')
checkpoint_f = os.path.join(checkpoint_dir, 'model.pt')
training_losses = []
valid_fde = []
valid_ade = []
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
    
epoch = 0
epochs = args.num_epochs
#define model 
if args.model_type == "CNN":
    print("making CNN")
    model = CNN(obs_seq_len,pred_seq_len).to(device)
elif args.model_type == "LSTM":
    print("making lstm")

else:
    print("making GCNN")
criterion = nll()
optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)


#helper methods 


    #list of predictions is [batch size, steps - 8, ]    
def pred_to_samples(num_samples, pred):
    sx = torch.exp(pred[2]) #sx
    sy = torch.exp(pred[3]) #sy
    corr = torch.tanh(pred[4]) #corr
    sxsy = sx*sy
    cov = torch.FloatTensor([[sx*sx, corr*sxsy],[corr*sxsy, sy*sy]])
    means = torch.FloatTensor([pred[0],pred[1]])
    dist = torch.distributions.multivariate_normal.MultivariateNormal(means, cov)
    return dist.sample((num_samples,))
def ade_dist(pred,target):
    return (torch.sum(torch.square(pred-target))/pred.shape[1]).item() #assume the middle dimension
def fde_dist(pred, target):
    return (torch.sum(torch.square(pred[-1] - target[-1]))).item() #only compare the final location of pred to final location of tgt
def store_model(epoch):
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_f)
def load_model():
    global epoch, model, optimzier
    checkpoint = torch.load(checkpoint_f)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.laod_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']

    model.train() #resume training 
def load_losses(fdir):
    global training_losses, valid_ade, valid_fde
    with open(os.path.join(checkpoint_dir,"trainloss.pkl"), 'rb') as f:
        training_losses = pickle.load(f)
    with open(os.path.join(checkpoint_dir ,"valid_ade.pkl"),'rb') as f:
        valid_ade = pickle.load(f)
    with open(os.path.join(checkpoint_dir ,"valid_fde.pkl"), 'rb') as f:
        valid_fde =  pickle.load(f)   
def store_losses():
    with open(os.path.join(checkpoint_dir,"trainloss.pkl"),'wb') as f:
        pickle.dump(training_losses, f)
    with open(os.path.join(checkpoint_dir ,"valid_ade.pkl")) as f:
        pickle.dump(valid_ade,f)
    with open(os.path.join(checkpoint_dir ,"valid_fde.pkl"), 'wb') as f:
        pickle.dump(valid_fde,f)






if args.load_model:
    print("loading model... \n")
    load_model()
    load_losses()
else:
    print("not loading model \n")

#start training loop
def train():
    model.train()
    train_loss = 0
    for idx, (inputs, labels) in enumerate(train_dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        preds = model(inputs)
        loss  = criterion(preds, labels)
        loss.backward()
        #clip gradient here
        optimizer.step()
        train_loss +=loss.item()
    train_batch_loss = train_loss/len(train_dataloader)
    training_losses.append(train_batch_loss)
def valid():
    model.eval()
    with torch.no_grad():
        batch_sum_ade = 0
        batch_sum_fde=0
        for idx, (inputs, labels) in enumerate(test_dataloader): #batch size, steps, etc
            inputs = inputs.to(device)
            labels = labels.to(device)
            T = inputs.shape[0]
            N = inputs.shape[1]
            preds = model(inputs) #each step has a distribution. So take a sample of each step and then turn it into something cumulative

            for b in range(T): #iterate over batches
                curr = preds[b,:,:]
                #iterate over curr
                num_samples = 20
                num_steps = curr.shape[0]
                sample_list = []
                for i in range(num_steps):
                    sample_list.append(pred_to_samples(num_samples, curr[i]))
                    #now should have a list of samples for each step

                samples = torch.stack(samples,dim=1) #stack it along middle dim to get [num_samples, num_steps, (x,y)]
                #now cumsum it 
                abs_samples = samples.cumsum(dim=1) #dim =1 since it is the number of steps shape is [num_samples, steps, rel pos]
                # now take the minimum ade/fde of the absolute samples to it 
                fde_dists = []
                ade_dists = []
                for i in range(num_samples):
                    #calculate the metrics for each
                    fde_dists.append(fde_dist(abs_samples[i]))
                    ade_dists.append(ade_dist(abs_samples[i]))
                #now take the min
                min_fde = min(fde_dists)
                min_ade = min(ade_dists)
                batch_sum_ade+=min_ade
                batch_sum_fde+=min_fde
            
            
        batch_sum_ade/= len(test_dataloader)
        batch_sum_fde/= len(test_dataloader)
        valid_ade.append(batch_sum_ade)
        valid_fde.append(batch_sum_fde)

for e in range(epoch,epochs):
    train()
    valid()
    store_model(e)
    store_losses()

