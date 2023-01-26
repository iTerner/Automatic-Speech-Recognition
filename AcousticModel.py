import os
import torch
import torchaudio
from tqdm import tqdm
import torchaudio.transforms as T
from functools import partial
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import json
from dataset import TIDIGITSDataset

def wav2vec2_collate(batch, max_pred:int) -> tuple:
    r"""
    The function convert the given batch to be in the format for the training process

    ### Args
    - batch (list) - The given batch
    - max_pred (int) - The maximum number of prediction 

    ### Return
    All the neccesary things for the training
    """
    data = [e[0].squeeze(0) for e in batch]
    input_lengths = torch.tensor([e[0].shape[1] for e in batch])

    # pad the data
    padded_data = torch.zeros((len(data), max(input_lengths)))
    for i in range(len(data)):
        padded_data[i, :input_lengths[i]] = data[i]
    
    target = [e[1] for e in batch]
    target_lengths = torch.tensor([len(e[1]) for e in batch])

    zer = torch.zeros((len(target), max(target_lengths), target[0].shape[1]))
    for i in range(len(target)):
        zer[i, :target_lengths[i]] = target[i]
        zer[i, target_lengths[i]:] = target[i][-1]

    return padded_data, zer, input_lengths, target_lengths


def validate(model:nn.Module, loader:DataLoader, criterion:nn, device:str) -> float:
    r"""
    The function evaluate the given model on the given data
    
    ### Args
    - model (nn.Module) - The given model
    - loader (DataLoader) - The given dataset
    - criterion (nn) - The loss function
    - device (str) - The device we are working on ("cpu" or "cuda")

    ### Return 
    The validation loss
    """
    model.eval()
    model.to(device)

    val_losses = []
    with torch.no_grad():
        for data, target, input_lengths, target_lengths in tqdm(loader):
            data, target, input_lengths = data.to(device), target.to(device), input_lengths.to(device)
            target = target.argmax(dim=-1)

            emissions, lengths = model(data, input_lengths)
            emissions = emissions.log_softmax(dim=-1)

            loss = criterion(emissions.transpose(1, 0), target, lengths, target_lengths)
            val_losses.append(loss.item())

    avg_loss = sum(val_losses) / len(val_losses)
    print(f"validation loss: {avg_loss}")
    return avg_loss

def train(model:nn.Module, train_loader:DataLoader, val_loader:DataLoader, optimizer:torch.optim, criterion:nn, args:dict, device:str, save_path:str=""):
    r"""
    The function trains the given model on the given dataset using the given arguments

    ### Args
    - model (nn.Module) - The given model
    - train_loader (DataLoader) - The train dataloader
    - val_loader (DataLoader) - The validation dataloader
    - optimizer (torch.optim) - The optimizer
    - criterion (nn) - The loss function
    - args (dict) - The training arguments
    - save_path (str) - The save path (default "")

    ### Return
    """
    model.to(device)
    model.train()

    num_epochs = args["num_epochs"]
    train_losses, val_losses = [], []
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        for data, target, input_lengths, target_lengths in tqdm(train_loader):
            data, target, input_lengths = data.to(device), target.to(device), input_lengths.to(device)
            target = target.argmax(dim=-1)

            # check if the target contains zeros
            if torch.sum(target == 0) > 0:
                print("traget contains zeros!")

            optimizer.zero_grad()
            emissions, lengths = model(data, input_lengths)
            emissions = emissions.log_softmax(dim=-1)
            
            loss = criterion(emissions.transpose(1, 0), target, lengths, target_lengths)
            epoch_losses.append(loss.item())
            loss.backward()
            optimizer.step()

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        train_losses.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}] | loss: {avg_loss}")

        if args["save_model"]:
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss
            }
            torch.save(checkpoint, os.path.join(save_path, "checkpoint.pt"))

        if val_loader is not None:
            val_loss = validate(model, val_loader, criterion, device)
            val_losses.append(val_loss)
        

def main(args:dict):
    r"""
    The main funciton

    ### Args
    - args (dict) - all the training arguments
    """
    # setting random seed 
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"working on {device}")

    # load wav2vec2 
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H

    # need to apply resample since the dataset sample rate is different than WAV2VEC2's training data sample rate
    transforms = T.Resample(20000, bundle.sample_rate)

    # create and split the dataset to train and validation
    dataset = TIDIGITSDataset(root="", data_type="train", transforms=transforms)
    # split to 80% train and 20% validation
    train_size = int(0.8 * len(dataset))
    train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
    collate_fn = partial(wav2vec2_collate, max_pred=args["max_pred"])

    # define the train and validation loaders
    train_loader = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args["batch_size"], shuffle=True, collate_fn=collate_fn)

    # save the train and validation datasets for later use
    data_save_path = os.path.join("datasets pytorch")
    if not os.path.exists(data_save_path):
        os.makedirs(data_save_path)

    torch.save(train_dataset, os.path.join(data_save_path, "train_dataset.pt"))
    torch.save(val_dataset, os.path.join(data_save_path, "validation_dataset.pt"))
    
    # we used WAV2VEC2 as the acoustic model, so first of all, we load the model
    model = bundle.get_model()
    model.aux = nn.Linear(model.aux.in_features, len(dataset.letter2index))
    # we finetune the model. for that, we need to freeze all the layers except the last one
    for param in model.parameters():
        param.requires_grad = False

    for param in model.aux.parameters():
        param.requires_grad = True

    # optimizer and loss function
    # optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"], weight_decay=0.05)
    optimizer = torch.optim.Adam(model.parameters(), lr=args["lr"])
    criterion = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    # train the model
    save_path = os.path.join("checkpoints")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    train(model, train_loader, val_loader, optimizer, criterion, args, device, save_path)

    with open(os.path.join(save_path, "training_args.json"), "w") as f:
        json.dump(args, f)

    # preform 1 validation run on the final model
    validate(model, val_loader, criterion, device)

if __name__ == "__main__":
    args = {
        "batch_size": 64,
        "num_epochs": 10,
        "lr": 0.001,
        "max_pred": 64,
        "save_model": True
    }
    main(args)