import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from transformers import Trainer

from utils import compute_metrics, set_seed
from sklearn.metrics import accuracy_score, f1_score

import numpy as np
import pickle
import sklearn.linear_model as linear_model
from sklearn import model_selection

import os

os.environ["WANDB_DISABLED"] = "true"

@torch.no_grad()
def evaluation(model, data_loader, device = None):
    print(data_loader)
    trainer = Trainer(
        model=model,
        eval_dataset=data_loader,
        compute_metrics=compute_metrics
        )

    res = trainer.evaluate()

    return {'Acc': res['eval_accuracy'], 'F1': res['eval_f1']}

@torch.no_grad()
def evaluation_cnn(model, dataset, device):
    
    model.eval()

    data_loader = DataLoader(dataset, batch_size=16, shuffle=False)

    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():

        for inputs, labels in data_loader:

            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')

    return {'Acc': acc, 'F1': f1}

def compute_losses_binary(net, loader, path, label, device, save, type):
    criterion = nn.CrossEntropyLoss(reduction="none")
    all_losses = []
    all_logits = []
    all_labels = []

    for batch in loader:
        if type == "cnn":
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            logits = net(inputs)
        else: 
            inputs = batch['input_values']
            targets = batch['labels']
            inputs, targets = inputs.to(device), targets.to(device)

            logits = net(inputs).logits

        losses = criterion(logits, targets).cpu().detach().numpy()
        #losses = criterion(logits, targets).numpy(force=True)
        for l in losses:
            all_losses.append(l)

        all_logits.append(logits.cpu().detach().numpy())
        all_labels.append(targets.cpu().detach().numpy())
        
    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    if save:
        with open(path + f'{label}_output_labels.pkl', 'wb') as f:
            pickle.dump(all_logits, f)
        with open(path + f'{label}_labels.pkl', 'wb') as f:
            pickle.dump(all_labels, f)

    return np.array(all_losses)

def simple_mia(sample_loss, members, n_splits=10, random_state=0):
    unique_members = np.unique(members)
    if not np.all(unique_members == np.array([0, 1])):
        raise ValueError("members should only have 0 and 1s")

    attack_model = linear_model.LogisticRegression()
    cv = model_selection.StratifiedShuffleSplit(
        n_splits=n_splits, random_state=random_state
    )
    return model_selection.cross_val_score(
        attack_model, sample_loss, members, cv=cv, scoring="accuracy"
    )

def cal_mia(model, forget_dataloader_test, test_dataloader, path, device, save, type):
    set_seed(42)

    forget_losses = compute_losses_binary(model, forget_dataloader_test, path, "forget", device=device, save=save, type=type)
    unseen_losses = compute_losses_binary(model, test_dataloader, path, "test", device=device, save=save, type=type)

    print(forget_losses.shape, unseen_losses.shape)

    if save: 
        # save in a pickle file
        with open(path + 'forget_losses.pkl', 'wb') as f:
            pickle.dump(forget_losses, f)
        with open(path + 'test_losses.pkl', 'wb') as f:
            pickle.dump(unseen_losses, f)

    if len(forget_losses) > len(unseen_losses):
        np.random.shuffle(forget_losses)
        forget_losses = forget_losses[: len(unseen_losses)]
    elif len(forget_losses) < len(unseen_losses):
        np.random.shuffle(unseen_losses)
        unseen_losses = unseen_losses[: len(forget_losses)]
    
    print(forget_losses.shape, unseen_losses.shape)

    samples_mia = np.concatenate((unseen_losses, forget_losses)).reshape((-1, 1))
    labels_mia = [0] * len(unseen_losses) + [1] * len(forget_losses)

    # shuffle the data
    indices = np.arange(len(samples_mia))
    np.random.shuffle(indices)
    samples_mia = samples_mia[indices]
    labels_mia = np.array(labels_mia)[indices]

    mia_scores = simple_mia(samples_mia, labels_mia)
    forgetting_score = abs(0.5 - mia_scores.mean())

    return {'MIA': mia_scores.mean(), 'Forgeting Score': forgetting_score}

def print_evaluation_metrics(model: nn.Module, forget_dataset: Dataset, val_dataset: Dataset, test_dataset: Dataset, save_path: str, device, save = True, bs = 4, external_seed = 0, type = ""):

    # Performance
    print("Validation on val set") 
    val_acc = evaluation(model, val_dataset) if type != "cnn" else evaluation_cnn(model, val_dataset, device)
    print("Validation on test set")
    test_acc = evaluation(model, test_dataset) if type != "cnn" else evaluation_cnn(model, test_dataset, device)
    print("Validation on forget set")
    forget_acc = evaluation(model, forget_dataset) if type != "cnn" else evaluation_cnn(model, forget_dataset, device)

    print("Evaluation of MIA")
    new_path = save_path + "mia_outputs/" if external_seed == 0 else save_path + "mia_outputs_seed_" + str(external_seed) + "/"
    os.makedirs(new_path, exist_ok=True) if save else None

    forget_dataloader = DataLoader(forget_dataset, batch_size=bs, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=bs, shuffle=False)

    mia = cal_mia(model=model, forget_dataloader_test=forget_dataloader, test_dataloader=test_dataloader, path=new_path, device=device, save=save, type=type)
    print(f'Test Acc: {val_acc}')
    print()
    print(f'Unseen Acc: {test_acc}')
    print()
    print(f'Forget Acc: {forget_acc}')
    print()
    print(f'MIA: {mia}')
    print()
    print(f'Final Score: {(val_acc["Acc"] + (1 - abs(mia["MIA"] - 0.5) * 2)) / 2}')
    print()
    dict = {'Test Acc': val_acc, 'Unseen Acc': test_acc, 'Forget acc': forget_acc, 'MIA': mia, 'Final Score': ((val_acc["Acc"] + (1 - abs(mia["MIA"] - 0.5) * 2)) / 2)}
    return dict