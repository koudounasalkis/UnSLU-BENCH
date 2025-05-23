import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import random

from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import pandas as pd

import argparse

""" Define Command Line Parser """
def parse_cmd_line_params():
    parser = argparse.ArgumentParser(description="Unlearning Script")
    parser.add_argument(
        "--batch",
        help="batch size",
        default=8, 
        type=int,
        required=False)
    parser.add_argument(
        "--feature_extractor_checkpoint",
        help="feature extractor to use",
        default="facebook/wav2vec2-base",  #
        type=str,                          
        required=False) 
    parser.add_argument(
        "--model_name_or_path",
        help="model to use",
        default="speech_unlearning/models/slurp/w2v2-base",  
        type=str,                          
        required=False)                     
    parser.add_argument(
        "--df_train",
        help="path to the train df",
        default="data_name/train.csv",
        type=str,
        required=False) 
    parser.add_argument(
        "--df_val",
        help="path to the val df",
        default="data_name/val.csv",
        type=str,
        required=False) 
    parser.add_argument(
        "--df_test",
        help="path to the test df",
        default="data_name/test.csv",
        type=str,
        required=False)
    parser.add_argument(
        "--df_retain",
        help="path to the retain df",
        default="data_name/retain.csv",
        type=str,
        required=False)
    parser.add_argument(
        "--df_forget",
        help="path to the forget df",
        default="data_name/forget.csv",
        type=str,
        required=False)
    parser.add_argument(
        "--unlearner",
        help="unlearner to use",
        default="None",
        type=str,
        required=False)
    parser.add_argument(
        "--output_dir",
        help="path to the output directory",
        default="models/slurp",
        type=str,
        required=False)
    parser.add_argument(
        "--lr",
        help="learning rate",
        default=1e-4,
        type=float,
        required=False)
    parser.add_argument(
        "--max_duration",
        help="Maximum audio duration",
        default=10.0,
        type=float,
        required=False)
    parser.add_argument(
        "--use_bad_teaching",
        default=1,
        type=int,
        required=False)
    parser.add_argument(
        "--seed",
        help="seed",
        default=0,
        type=int,
        required=False)
    parser.add_argument(
        "--unfreeze_encoder_layer", 
        help="unfreeze encoder layer",
        default=-1,
        type=int,
        required=False)
    parser.add_argument(
        "--epochs",
        help="epochs",
        default=1,
        type=int,
        required=False
    )
    parser.add_argument(
        "--dataset",
        help="Dataset to use, choose between [slurp, fsc, italic, de-DE, fr-FR]",
        default="forget",
        type=str,
        required=False
    )
    args = parser.parse_args()
    return args

""" Define Metric """
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average='macro')

    print('Accuracy: ', acc)
    print('F1 Macro: ', f1_macro)
    
    return { 'eval_accuracy': acc, 'eval_f1': f1_macro }

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

""" Read and Process Data"""
def read_data(df_train_path, df_val_path):
    df_train = pd.read_csv(df_train_path, index_col=None)
    df_val = pd.read_csv(df_val_path, index_col=None)
    
    ## Prepare Labels
    labels = df_train['intent'].unique()
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label
    num_labels = len(id2label)

    ## Train
    for index in range(0,len(df_train)):
        df_train.loc[index,'label'] = label2id[df_train.loc[index, 'intent']]
    df_train['label'] = df_train['label'].astype(int)

    ## Validation
    for index in range(0,len(df_val)):
        df_val.loc[index,'label'] = label2id[df_val.loc[index, 'intent']]
    df_val['label'] = df_val['label'].astype(int)

    print("Label2Id: ", label2id)
    print("Id2Label: ", id2label)
    print("Num Labels: ", num_labels)

    return df_train, df_val, num_labels, label2id, id2label, labels

""" Define model and feature extractor """
def define_model(
    model_checkpoint, 
    num_labels, 
    label2id, 
    id2label, 
    feature_extractor_checkpoint=None,
    device="cuda"
    ):
    print("Model checkpoint: ", model_checkpoint)
    print("Feature extractor checkpoint: ", feature_extractor_checkpoint)
    feature_extractor = AutoFeatureExtractor.from_pretrained(feature_extractor_checkpoint) if feature_extractor_checkpoint else AutoFeatureExtractor.from_pretrained(model_checkpoint)
    model = AutoModelForAudioClassification.from_pretrained(
        model_checkpoint, 
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label, 
        ignore_mismatched_sizes=True
        ).to(device)
    return feature_extractor, model