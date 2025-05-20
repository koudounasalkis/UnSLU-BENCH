import torch
import librosa
import numpy as np

""" Dataset Class """
class Dataset_slurp_fsc(torch.utils.data.Dataset):
    def __init__(self, examples, feature_extractor, max_duration, dataset, augmentation=False):
        self.examples = examples['path']
        self.labels = examples['label']
        self.feature_extractor = feature_extractor
        self.max_duration = max_duration
        self.augmentation = augmentation
        self.sr = 16_000

        self.default_path = f"datasets/{dataset}/"
        self.examples = [self.default_path + example for example in self.examples]

    def __getitem__(self, idx):

        ## Augmentation:
            # 1: Add noise
            # 2: Change speed up
            # 3: Change pitch
            # 4: Change speed down
            # 5: Add noise + Change speed (up) + Change pitch
            # 6: Add noise + Change speed (down) + Change pitch
        if self.augmentation:
            ### Augment or not, with a probability of 0.15
            augment = np.random.choice([True, False], p=[0.15, 0.85]) 
            ### Choose augmentation type
            augmentation_type = np.random.choice([1, 2, 3, 4, 5, 6])
            if augment:
                try:
                    audio, sr = librosa.load(self.examples[idx], sr=self.sr)
                    if augmentation_type == 1:
                        ### Add noise
                        noise = np.random.normal(0, 0.005, audio.shape[0])
                        audio = audio + noise
                    elif augmentation_type == 2:
                        ### Change speed up
                        audio = librosa.effects.time_stretch(audio, rate=1.2)
                    elif augmentation_type == 3:
                        ### Change pitch
                        audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=4)
                    elif augmentation_type == 4:
                        ### Change speed down
                        audio = librosa.effects.time_stretch(audio, rate=0.8)
                    elif augmentation_type == 5:
                        ### Add noise + Change speed (up) + Change pitch
                        noise = np.random.normal(0, 0.005, audio.shape[0])
                        audio = audio + noise
                        audio = librosa.effects.time_stretch(audio, rate=1.2)
                        audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=4)
                    elif augmentation_type == 6:
                        ### Add noise + Change speed (down) + Change pitch
                        noise = np.random.normal(0, 0.005, audio.shape[0])
                        audio = audio + noise
                        audio = librosa.effects.time_stretch(audio, rate=0.8)
                        audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=4)
                    ### Extract features
                    inputs = self.feature_extractor(
                        audio.squeeze(),
                        sampling_rate=self.feature_extractor.sampling_rate, 
                        return_tensors="pt",
                        max_length=int(self.feature_extractor.sampling_rate * self.max_duration), 
                        truncation=True,
                        padding='max_length')
                except:
                    print("Audio not available", self.examples[idx])

            else:
                try:
                    inputs = self.feature_extractor(
                        librosa.load(self.examples[idx], sr=self.sr)[0].squeeze(),
                        sampling_rate=self.feature_extractor.sampling_rate, 
                        return_tensors="pt",
                        max_length=int(self.feature_extractor.sampling_rate * self.max_duration), 
                        truncation=True,
                        padding='max_length')
                except:
                    print("Audio not available: ", self.examples[idx])
        else:
            try:
                inputs = self.feature_extractor(
                    librosa.load(self.examples[idx], sr=self.feature_extractor.sampling_rate)[0].squeeze(),
                    sampling_rate=self.feature_extractor.sampling_rate, 
                    return_tensors="pt",
                    max_length=int(self.feature_extractor.sampling_rate * self.max_duration), 
                    truncation=True,
                    padding='max_length')
            except:
                print("Audio not available", self.examples[idx])

        try:
            item = {'input_values': inputs['input_values'].squeeze(0)}
            item["labels"] = torch.tensor(self.labels[idx])
        except:
            item = { 'input_values': [], 'labels': [] }
        return item

    def __len__(self):
        return len(self.examples)

class Dataset_italic_sm(torch.utils.data.Dataset):
    def __init__(self, examples, feature_extractor, label2id, max_duration, device):
        self.examples = examples
        self.labels = [int(label2id[e]) for e in examples['intent']]
        self.feature_extractor = feature_extractor
        self.max_duration = max_duration
        self.device = device 

    def __getitem__(self, idx):
        inputs = self.feature_extractor(
            self.examples[idx]['audio']['array'],
            sampling_rate=self.feature_extractor.sampling_rate, 
            return_tensors="pt",
            max_length=int(self.feature_extractor.sampling_rate * self.max_duration), 
            truncation=True,
            padding='max_length'
        )
        item = {'input_values': inputs['input_values'].squeeze(0)}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.examples)

def get_forget_retain_datasets(ds_train, data_path):
    with open(data_path + 'forget_indexes.txt') as f:
        forget_indexes = f.readlines()
    forget_indexes = [int(x.strip()) for x in forget_indexes]

    with open(data_path + 'retain_indexes.txt') as f:
        retain_indexes = f.readlines()
    retain_indexes = [int(x.strip()) for x in retain_indexes]

    ds_forget = ds_train.select(forget_indexes)
    ds_retain = ds_train.select(retain_indexes)

    return ds_forget, ds_retain

def get_test_val_datasets(ds_val, data_path): 
    with open(data_path + 'test_indexes.txt') as f:
        test_indexes = f.readlines()
    test_indexes = [int(x.strip()) for x in test_indexes]

    with open(data_path + 'val_indexes.txt') as f:
        val_indexes = f.readlines()
    val_indexes = [int(x.strip()) for x in val_indexes]

    ds_val_new = ds_val.select(val_indexes)
    ds_test = ds_val.select(test_indexes)

    return ds_val_new, ds_test