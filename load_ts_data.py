def get_dataloader(fdir): #helper method to return dataloader
    l = os.listdir(fdir)
    datasets = [None]*len(l)
    for i in range(len(l)):
        temp = os.path.join(fdir,l[i])
        datasets[i] = TS_Data(temp)
    dataset =  torch.utils.data.ConcatDataset(datasets)
    dataloader = DataLoader(dataset, batch_size= 64)
    return dataloader

def main():
    fdir = "/home/warren/Documents/UGRA/AIM/csv_files"
    
    dataloader = get_dataloader(fdir)

if __name__ == "__main__":
    main()