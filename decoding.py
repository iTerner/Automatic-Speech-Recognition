import torch
import torch.nn as nn
import torchaudio
import json
import torchaudio.transforms as T
import os
from functools import partial
from torch.nn.utils.rnn import pad_sequence
from torchaudio.models.decoder import ctc_decoder
from torch.utils.data import DataLoader
from torchmetrics import WordErrorRate, CharErrorRate
from tqdm import tqdm
from dataset import TIDIGITSDataset


# taken form https://pytorch.org/audio/main/tutorials/asr_inference_with_ctc_decoder_tutorial.html#greedy-decoder
class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> list:
        """Given a sequence emission over labels, get the best path
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          List[str]: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        joined = "".join([self.labels[i.item()] for i in indices])
        return joined.replace("|", " ").strip().split()

def test_wav2vec2_collate(batch) -> tuple:
    r"""
    The function convert the given batch to be in the format for the training process

    ### Args
    - batch (list) - The given batch
    - max_pred (int) - The maximum number of prediction 

    ### Return
    All the neccesary things for the training
    """
    data = [e[0].squeeze(0) for e in batch]
    input_lengths = torch.tensor([e[0].squeeze(0).shape[0] for e in batch])
    filenames = [e[1] for e in batch]

    # pad the data
    padded_data = torch.zeros((len(data), max(input_lengths)))
    for i in range(len(data)):
        padded_data[i, :input_lengths[i]] = data[i]

    return padded_data, input_lengths, filenames

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


def decode():
    # setting random seed 
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # load the model
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H

    # load train dataset
    transforms = T.Resample(20000, bundle.sample_rate)
    dataset = TIDIGITSDataset(root="", data_type="train", transforms=transforms)
    mapping_dict = dataset.mapping_dict
    letter2index = dataset.letter2index
    index2letter = {v: k for k, v in letter2index.items()}

    # load the model - continue
    model = bundle.get_model()
    model.aux = nn.Linear(model.aux.in_features, len(letter2index))
    model.load_state_dict(torch.load(os.path.join("checkpoints", "checkpoint.pt"), map_location=torch.device("cpu"))["model_state_dict"])
    model.eval()

    # load the training arguments
    training_args = json.load(open(os.path.join("checkpoints", "training_args.json")))
    
    # get all the possible tokens
    tokens = [index2letter[i].lower() for i in range(len(index2letter))]

    # define train and validation loaders
    collate_func = partial(wav2vec2_collate, max_pred=training_args["max_pred"])
    trained_train_dataset = torch.load(os.path.join("datasets pytorch", "train_dataset.pt"))
    trained_val_dataset = torch.load(os.path.join("datasets pytorch", "validation_dataset.pt"))
    train_loader = DataLoader(trained_train_dataset, batch_size=1, shuffle=False, collate_fn=collate_func)
    val_loader = DataLoader(trained_val_dataset, batch_size=1, shuffle=False, collate_fn=collate_func)
    
    wer, cer = WordErrorRate(), CharErrorRate()
    for loader, name in zip([train_loader, val_loader], ["train", "validation"]):
        print(f"===================={name}=====================")
        
        for beam_size in [1, 50, 500]:
            print(f"====================Beam size: {beam_size}=====================")
            
            # creating all the decoder
            greedy_decoder = GreedyCTCDecoder(tokens)
            ctc_decoder_with_lm = ctc_decoder("lexicon.txt", tokens=tokens, lm="lang_model.arpa", sil_token=dataset.sil_token, blank_token=dataset.blank_token, beam_size=beam_size)
            ctc_decoder_without_lm = ctc_decoder("lexicon.txt", tokens=tokens, sil_token=dataset.sil_token, blank_token=dataset.blank_token, beam_size=beam_size)
            
            greedy_wer, with_lm_wer, without_lm_wer = [], [], []
            greedy_cer, with_lm_cer, without_lm_cer = [], [], []

            for data, target, input_length, target_length in tqdm(loader):
                with torch.no_grad():
                    emission, _ = model(data)
                    
                    # decode
                    greedy_result = (greedy_decoder(emission[0]))
                    ctc_decoder_with_lm_results = (ctc_decoder_with_lm(emission))
                    ctc_decoder_without_lm_results = (ctc_decoder_without_lm(emission))

                    # get the prediction from each decoder, in addition to the correct one
                    with_lm_res = [result[0].words for result in ctc_decoder_with_lm_results]
                    without_lm_res = [result[0].words for result in ctc_decoder_without_lm_results]
                    correct = ["".join(list(map(lambda x: index2letter[x.item()], t))) for t in target.argmax(-1)]
                    greedy_pred = " ".join(greedy_result).upper()
                    with_lm_pred = " ".join(mapping_dict[x] for x in with_lm_res[0])
                    without_lm_pred = " ".join(mapping_dict[x] for x in without_lm_res[0])
                    correct_pred = (correct[0]).replace(dataset.sil_token, " ")

                    # compute the WER and CER
                    greedy_wer.append(wer(greedy_pred, correct_pred))
                    greedy_cer.append(cer(greedy_pred, correct_pred))
                    with_lm_wer.append(wer(with_lm_pred, correct_pred))
                    with_lm_cer.append(cer(with_lm_pred, correct_pred))
                    without_lm_wer.append(wer(without_lm_pred, correct_pred))
                    without_lm_cer.append(cer(without_lm_pred, correct_pred))

            print(f"greedy WER: {sum(greedy_wer) / len(greedy_wer)}, greedy CER: {sum(greedy_cer) / len(greedy_cer)}")
            print(f"CTC with LM WER: {sum(with_lm_wer) / len(with_lm_wer)}, CTC with LM CER: {sum(with_lm_cer) / len(with_lm_cer)}")
            print(f"CTC without LM WER: {sum(without_lm_wer) / len(without_lm_wer)}, CTC without LM CER: {sum(without_lm_cer) / len(without_lm_cer)}")

    # load test data
    test_collate_func = partial(test_wav2vec2_collate)
    test_dataset = TIDIGITSDataset(root="", data_type="test", transforms=transforms)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=test_collate_func)

    # decode the test dataset with the best decoder, CTC decoder without lm with beam size 50
    best_decoder = ctc_decoder("lexicon.txt", tokens=tokens, sil_token=dataset.sil_token, blank_token=dataset.blank_token, beam_size=50)
    with open("output.txt", "w") as f:
        for data, input_length, filename in tqdm(test_loader):
            emission, _ = model(data)
            res = best_decoder(emission)
            pred = "".join([result[0].words for result in res][0])
            f.writelines([f"{filename[0]} - {pred}\n"])
            

if __name__ == "__main__":
    decode()