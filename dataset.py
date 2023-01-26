import os
import torch
import torchaudio
from torch.utils.data import Dataset

class TIDIGITSDataset(Dataset):
    def __init__(self, root:str, data_type:str, transforms:torchaudio.transforms=None):
        r"""
        ### Args
        - root (str) - The root path
        - data_type (str) - The type of the data ("train" or "test")
        - transforms (torchaudio.transforms) - Used transforms for the data (default None)
        """
        self.root = os.path.join(root, data_type)
        self.data_type = data_type
        self.transforms = transforms

        self.mapping_dict = {
            "1": "ONE",
            "2": "TWO",
            "3": "THREE",
            "4": "FOUR",
            "5": "FIVE",
            "6": "SIX",
            "7": "SEVEN",
            "8": "EIGHT",
            "9": "NINE",
            "z": "ZERO",
            "o": "OH"
        }

        self.sil_token = "|"
        self.blank_token = "-"

        # map all the unique letters from the train_transcription
        all_letters = list(f"{self.blank_token}efghinorstuvwxz{self.sil_token}".upper())
        self.letter2index = {letter: i for i, letter in enumerate(all_letters)}

        self.data, self.labels, self.filenames = [], [], []
        for root, dirs, files in os.walk(self.root):
            for file_name in files:
                if ".wav" in file_name:
                    self.data.append(os.path.join(root, file_name))
                    if self.data_type == "train":
                        self.labels.append(self.map_label(file_name.split(".")[0][:-1])) # ignore the last letter
                    elif self.data_type == "test":
                        self.filenames.append(file_name)

    
    def map_label(self, label:str) -> str:
        r"""
        The function maps the label 

        ### Args 
        - label (str) - the given label

        ### Return
        The mapped label
        """
        mapped_label = ""
        for i in range(len(label)):
            if i > 0:
                mapped_label += " "
            mapped_label += self.mapping_dict[label[i]]

        return mapped_label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        data, sample_rate = torchaudio.load(self.data[idx])

        # apply transform 
        if self.transforms:
            data = self.transforms(data)

        if self.data_type == "train":
            labels = self.labels[idx]

            labels_tensor = torch.zeros((len(labels), len(self.letter2index)))
            for i in range(len(labels)):
                labels_tensor[i, self.letter2index[labels[i].replace(" ", self.sil_token)]] = 1

            return data, labels_tensor
        elif self.data_type == "test":
            return data, self.filenames[idx]
