import torch.nn as nn
import torchaudio
import torch
from torch.nn.utils.rnn import pad_sequence

# DT2119, Lab 4 End-to-end Speech Recognition

# Variables to be defined --------------------------------------
''' 
train-time audio transform object, that transforms waveform -> spectrogram, with augmentation
''' 
train_audio_transform = nn.Sequential(
    torchaudio.transforms.MelSpectrogram(n_mels=80),
    torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
    torchaudio.transforms.TimeMasking(time_mask_param=35)
)
'''
test-time audio transform object, that transforms waveform -> spectrogram, without augmentation 
'''
test_audio_transform = nn.Sequential(
    torchaudio.transforms.MelSpectrogram(n_mels=80)
)

# Functions to be implemented ----------------------------------

def strToInt(text):
    result = []
    for char in text:
        if char == "'":
            result.append(0)
        elif char == ' ':
            result.append(1)
        else:
            result.append(ord(char) - ord('a') + 2)
    return result

def intToStr(labels):
    result = ""
    for label in labels:
        if label == 0:
            result += "'"
        elif label == 1:
            result += " "
        elif 2 <= label <= 27:
            result += chr(label - 2 + ord('a'))
    return result


def dataProcessing(data, transform):
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []

    for waveform, sample_rate, utterance, *_ in data:
        # Step 1: Apply audio transform
        spec = transform(waveform).squeeze(0).transpose(0, 1)  # shape: Time x Mel
        spectrograms.append(spec)
        input_lengths.append(spec.shape[0] // 2)  # halve due to CTC downsampling

        # Step 2: Convert utterance to label sequence
        label = torch.Tensor(strToInt(utterance.lower())).int()
        labels.append(label)
        label_lengths.append(len(label))

    # Step 3: Pad all sequences
    spectrograms = pad_sequence(spectrograms, batch_first=True).unsqueeze(1).transpose(2, 3)
    labels = pad_sequence(labels, batch_first=True)

    return spectrograms, labels, input_lengths, label_lengths
    
def greedyDecoder(output, blank_label=28):
    decoded_batch = []

    # output: B x T x C → B x T (take argmax over characters)
    max_probs = torch.argmax(output, dim=2)

    for batch in max_probs:
        decoded = []
        previous = blank_label
        for label in batch:
            if label.item() != previous and label.item() != blank_label:
                decoded.append(label.item())
            previous = label.item()
        decoded_batch.append(intToStr(decoded))

    return decoded_batch


def levenshteinDistance(ref,hyp):
    '''
    calculate levenshtein distance (edit distance) between two sequences
    arguments:
        ref: reference sequence
        hyp: sequence to compare against the reference
    output:
        edit distance (int)
    '''
