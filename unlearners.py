import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


from tqdm import tqdm
import time
import copy 

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import numpy as np

from utils import set_seed


def finetune(model, retain_dataloader, forget_loader, device, num_epochs=1, lr=5e-5, seed=0, type=""):

    print("Starting Fine-tuning")

    set_seed(seed)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    tic = time.time()
    for epoch in range(num_epochs):
        running_loss = 0
        total_samples = 0

        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for batch in tqdm(retain_dataloader):

            if type == "cnn":
                x_retain, y_retain = batch
            else:
                x_retain = batch['input_values']
                y_retain = batch['labels']
            
            x_retain = x_retain.to(device)
            y_retain = y_retain.to(device)

            # Classification Loss
            outputs_retain = model(x_retain)
            classification_loss = criterion(outputs_retain if type == "cnn" else outputs_retain.logits, y_retain)

            optimizer.zero_grad()
            classification_loss.backward()
            optimizer.step()

            running_loss += classification_loss.item() * x_retain.size(0)
            total_samples += x_retain.size(0)

        average_epoch_loss = running_loss / total_samples
        print(f"Epoch [{epoch+1}/{num_epochs}] - Total Loss: {running_loss:.4f}, Average Loss: {average_epoch_loss:.4f}")

    total_time = time.time() - tic
    

    return total_time

def cf_k(model, retain_dataloader, forget_dataloader, device, num_epochs=1, lr=5e-5, freezed_layers = 1, seed=0, type="", unfreezed_encoder_layer = -1):
        
    print(f"Starting CF-{freezed_layers}")

    set_seed(seed)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # freeze all layers except the last n ones
    for i, layer in enumerate(list(model.children())):
        if i >= len(list(model.children())) - freezed_layers:
            for param in layer.parameters():
                param.requires_grad = True
            print(f"unfreezed layer - {layer}")
        else:
            for param in layer.parameters():
                param.requires_grad = False
    
    if unfreezed_encoder_layer != -1:
        # unfreeze the encoder layer, only for Hubert model
        for param in model.hubert.encoder.layers[unfreezed_encoder_layer].parameters():
            param.requires_grad = True
        for i, layer in enumerate(model.hubert.encoder.layers):
            # print if the layer is frozen or not
            for param in layer.parameters():
                print(f"Layer {i} - {param.requires_grad}")
                break

    tic = time.time()

    for epoch in range(num_epochs):
        running_loss = 0

        for batch in tqdm(retain_dataloader):
            
            if type == "cnn":
                x_retain, y_retain = batch
            else:
                x_retain = batch['input_values']
                y_retain = batch['labels']

            x_retain = x_retain.to(device)
            y_retain = y_retain.to(device)

            # Classification Loss
            outputs_retain = model(x_retain)
            classification_loss = criterion(outputs_retain if type == "cnn" else outputs_retain.logits, y_retain)

            optimizer.zero_grad()
            classification_loss.backward()
            optimizer.step()

            running_loss += classification_loss.item() * x_retain.size(0)

        average_epoch_loss = running_loss / (len(retain_dataloader) * x_retain.size(0))
        print(f"Epoch [{epoch+1}/{num_epochs}] - Total Loss: {running_loss:.4f}")

    total_time = time.time() - tic
    
    return total_time

def neggrad(model, retain_dataloader, forget_dataloader, device, num_epochs=1, lr=5e-5, seed=0, type=""):

    print("Starting NegGrad")

    set_seed(seed)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    tic = time.time()
    for epoch in range(num_epochs):
        running_loss = 0

        for forget_batch in tqdm(forget_dataloader):

            if type == "cnn":
                x_forget, y_forget = forget_batch
            else:
                x_forget = forget_batch['input_values']
                y_forget = forget_batch['labels']

            outputs_forget = model(x_forget.to(device))
            loss_ascent_forget = -criterion(outputs_forget if type == "cnn" else outputs_forget.logits, y_forget.to(device))

            optimizer.zero_grad()
            loss_ascent_forget.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss_ascent_forget.item()

        print(f"Epoch [{epoch+1}/{num_epochs}] - Total Loss: {running_loss:.4f}")
    
    total_time = time.time() - tic

    return total_time

def advancedneggrad(model, retain_dataloader, forget_dataloader, device, num_epochs=1, lr=5e-5, seed=0, type=""):

    print("Starting Advanced NegGrad")

    set_seed(seed)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    dataloader_iterator = iter(forget_dataloader)

    tic = time.time()
    for epoch in range(num_epochs):
        running_loss = 0

        for retain_batch in tqdm(retain_dataloader):

            if type == "cnn":
                x_retain, y_retain = retain_batch
            else:
                x_retain = retain_batch['input_values']
                y_retain = retain_batch['labels']

            x_retain = x_retain.to(device)
            y_retain = y_retain.to(device)

            try:
                forget_batch = next(dataloader_iterator)
                if type == "cnn":
                    x_forget, y_forget = forget_batch
                else:
                    x_forget = forget_batch['input_values']
                    y_forget = forget_batch['labels']

            except StopIteration:
                dataloader_iterator = iter(forget_dataloader)
                forget_batch = next(dataloader_iterator)
                if type == "cnn":
                    x_forget, y_forget = forget_batch
                else:
                    x_forget = forget_batch['input_values']
                    y_forget = forget_batch['labels']

            if x_forget.size(0) != x_retain.size(0):
                continue

            x_forget = x_forget.to(device)
            y_forget = y_forget.to(device)

            outputs_retain = model(x_retain)
            outputs_forget = model(x_forget)

            loss_ascent_forget = -criterion(outputs_forget if type == "cnn" else outputs_forget.logits, y_forget)
            loss_retain = criterion(outputs_retain if type == "cnn" else outputs_retain.logits, y_retain)

            # Overall loss
            joint_loss = loss_ascent_forget + loss_retain

            #print("joint loss :", joint_loss.item())
            optimizer.zero_grad()
            joint_loss.backward()
            optimizer.step()

            running_loss += joint_loss.item() 

        print(f"Epoch [{epoch+1}/{num_epochs}] - Total Loss: {running_loss:.4f}")

    total_time = time.time() - tic

    return total_time

class Noise(nn.Module):
    def __init__(self, batch_size, *dim):
        super().__init__()
        self.noise = nn.Parameter(torch.randn(batch_size, *dim), requires_grad=True)

    def forward(self):
        return self.noise

def unsir(model, retain_dataloader, forget_dataloader, device, batch_size, num_epochs_1 = 1, num_epochs_2 = 1, lr=5e-5, noise_lr = 5e-4, seed=0, type=""):

    print("Starting UNSIR")
    
    set_seed(seed)

    num_epochs_1 = 1
    num_epochs_2 = 1
    noise_lr = 0.01

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    train_epoch_losses = []

    tic = time.time()
    for epoch in range(num_epochs_1):
        running_loss = 0

        for retain_batch, forget_batch in tqdm(zip(retain_dataloader, forget_dataloader)):

            if type == "cnn":
                x_retain, y_retain = retain_batch
            else:
                x_retain = retain_batch['input_values']
                y_retain = retain_batch['labels']

            if type == "cnn":
                x_forget, y_forget = forget_batch
            else:
                x_forget = forget_batch['input_values']
                y_forget = forget_batch['labels']

            y_retain = y_retain.to(device)
            batch_size_forget = y_forget.size(0)

            if x_retain.size(0) != batch_size or x_forget.size(0) != batch_size:
                continue

            # Initialize the noise
            if type == "cnn":
                noise_dim = x_retain.size(1), x_retain.size(2), x_retain.size(3)
                noise = Noise(batch_size_forget, *noise_dim).to(device)
            else:
                noise_dim = x_retain.size(1)
                noise = Noise(batch_size_forget, noise_dim).to(device)
            noise_optimizer = torch.optim.Adam(noise.parameters(), lr=noise_lr)
            noise_tensor = noise()[:batch_size_forget]

            # Update the noise for increasing the loss value.
            for _ in range(5):
                outputs = model(noise_tensor)
                with torch.no_grad():
                    target_logits = model(x_forget.to(device))
                # Maximize the similarity between noise data and forget features.
                loss_noise = -F.mse_loss(outputs if type == "cnn" else outputs.logits, target_logits if type == "cnn" else target_logits.logits)

                # Backpropagate to update the noise.
                noise_optimizer.zero_grad()
                loss_noise.backward(retain_graph=True)
                noise_optimizer.step()

            # Train the model with noise and retain image
            noise_tensor = torch.clamp(noise_tensor, 0, 1).detach().to(device)
            outputs = model(noise_tensor.to(device))
            loss_1 = criterion(outputs if type == "cnn" else outputs.logits, y_retain)

            outputs = model(x_retain.to(device))
            loss_2 = criterion(outputs if type == "cnn" else outputs.logits, y_retain)

            joint_loss = loss_1 + loss_2

            optimizer.zero_grad()
            joint_loss.backward()
            optimizer.step()
            running_loss += joint_loss.item() * x_retain.size(0)

        average_train_loss = running_loss / (len(retain_dataloader) * x_retain.size(0))
        train_epoch_losses.append(average_train_loss)
        print(f"Epoch [{epoch+1}/{num_epochs_1}] - Train Loss: {average_train_loss:.4f}")

    # UNSIR-2

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(num_epochs_2):
        running_loss = 0

        for batch in tqdm(retain_dataloader):

            if type == "cnn":
                x_retain, y_retain = batch
            else:
                x_retain = batch['input_values']
                y_retain = batch['labels']

            y_retain = y_retain.to(device)

            # Classification Loss
            outputs_retain = model(x_retain.to(device))
            classification_loss = criterion(outputs_retain if type == "cnn" else outputs_retain.logits, y_retain)

            optimizer.zero_grad()
            classification_loss.backward()
            optimizer.step()

            running_loss += classification_loss.item() * x_retain.size(0)

        average_epoch_loss = running_loss / (len(retain_dataloader) * x_retain.size(0))
        print(f"Epoch [{epoch+1}/{num_epochs_2}] - Total Loss: {running_loss:.4f}")

    total_time = time.time() - tic
    
    return total_time

class DistillKL(nn.Module):
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss
    
class SCRUBTraining:
    def __init__(self, teacher, student, retain_dataloader, forget_dataloader, T, device, lr, seed=0, type=""):
        self.teacher = teacher
        self.student = student
        self.retain_dataloader = retain_dataloader
        self.forget_dataloader = forget_dataloader

        self.type = type

        self.device = device
        self.T = T
        self.lr = lr

        self.seed = seed

        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_div = DistillKL(T)
        self.criterion_kd = DistillKL(T)

        self.optimizer = optim.AdamW(student.parameters(), lr=lr)

    def train_epoch(self):
        self.student.train()
        self.teacher.eval()

        total_loss_retain = 0
        total_loss_forget = 0

        # Training with retain data.
        for batch in self.retain_dataloader:
            
            if self.type == "cnn":
                x_retain, y_retain = batch
            else:
                x_retain = batch['input_values']
                y_retain = batch['labels']

            x_retain, y_retain = x_retain.to(self.device), y_retain.to(self.device)

            # Forward pass: Student
            outputs_retain_student = self.student(x_retain)

            # Forward pass: Teacher
            with torch.no_grad():
                outputs_retain_teacher = self.teacher(x_retain)

            # Loss computation
            loss_cls = self.criterion_cls(outputs_retain_student if self.type == "cnn" else outputs_retain_student.logits, y_retain)
            loss_div_retain = self.criterion_div(outputs_retain_student if self.type == "cnn" else outputs_retain_student.logits, outputs_retain_teacher if self.type == "cnn" else outputs_retain_teacher.logits)

            loss = loss_cls + loss_div_retain

            # Update total loss and accuracy for retain data.
            total_loss_retain += loss.item()

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Training with forget data.
        for batch in self.forget_dataloader:

            if self.type == "cnn":
                x_forget, _ = batch
            else:
                x_forget = batch['input_values']

            x_forget = x_forget.to(self.device)

            # Forward pass: Student
            outputs_forget_student = self.student(x_forget)

            # Forward pass: Teacher
            with torch.no_grad():
                outputs_forget_teacher = self.teacher(x_forget)

            # We want to maximize the divergence for the forget data.
            loss_div_forget = -self.criterion_div(outputs_forget_student if self.type == "cnn" else outputs_forget_student.logits, outputs_forget_teacher if self.type == "cnn" else outputs_forget_teacher.logits)

            # Update total loss and accuracy for forget data.
            total_loss_forget += loss_div_forget.item()

            # Backward pass
            self.optimizer.zero_grad()
            loss_div_forget.backward()
            self.optimizer.step()

        # Print average loss and accuracy for the entire epoch
        avg_loss_retain = total_loss_retain / len(self.retain_dataloader)

        avg_loss_forget = total_loss_forget / len(self.forget_dataloader)

        print(f'Epoch Retain: Avg Loss: {avg_loss_retain:.4f}')
        print(f'Epoch Forget: Avg Loss: {avg_loss_forget:.4f}')

def scrub(model, retain_dataloader, forget_dataloader, device, num_epochs=1, T=4.0, lr=5e-5, seed=0, type=""):
    
    print("Starting SCRUB")

    set_seed(seed)

    teacher = copy.deepcopy(model)

    # Initialize and train
    scrub_trainer = SCRUBTraining(teacher, model, retain_dataloader, forget_dataloader, T, device, lr, seed=seed, type=type)

    tic = time.time()
        
    for epoch in range(num_epochs):
        scrub_trainer.train_epoch()
        print(f"Epoch {epoch+1} completed.")

    total_time = time.time() - tic
    
    return total_time

def UnlearnerLoss(output, labels, full_teacher_logits, unlearn_teacher_logits, KL_temperature):
    labels = torch.unsqueeze(labels, dim = 1)
    
    f_teacher_out = F.softmax(full_teacher_logits / KL_temperature, dim=1)
    u_teacher_out = F.softmax(unlearn_teacher_logits / KL_temperature, dim=1)

    # label 1 means forget sample
    # label 0 means retain sample
    overall_teacher_out = labels * u_teacher_out + (1-labels)*f_teacher_out
    student_out = F.log_softmax(output / KL_temperature, dim=1)
    return F.kl_div(student_out, overall_teacher_out, reduction='batchmean')

def unlearning_step(model, unlearning_teacher, full_trained_teacher, unlearn_data_loader, optimizer, 
            device, KL_temperature, type):
    losses = []
    for batch in unlearn_data_loader:
            
        x, y = batch
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            full_teacher_logits = full_trained_teacher(x) if type == "cnn" else full_trained_teacher(x).logits
            if unlearning_teacher is not None:
                unlearn_teacher_logits = unlearning_teacher(x) if type == "cnn" else full_trained_teacher(x).logits
            else: 
                unlearn_teacher_logits = torch.randn_like(full_teacher_logits)
        output = model(x) if type == "cnn" else model(x).logits
        optimizer.zero_grad()
        loss = UnlearnerLoss(output = output, labels=y, full_teacher_logits=full_teacher_logits, 
                unlearn_teacher_logits=unlearn_teacher_logits, KL_temperature=KL_temperature)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().numpy())
    return np.mean(losses)

def bad_teaching(model, unlearning_teacher, full_trained_teacher, retain_data, forget_data, device, epochs = 1,
                optimizer = 'adamW', lr = 5e-5, batch_size = 8, KL_temperature = 1, seed = 0, type = ""):
    # creating the unlearning dataset.
    tic = time.time()


    unlearning_data = UnLearningData(forget_data=forget_data, retain_data=retain_data, type=type)
    unlearning_loader = DataLoader(unlearning_data, batch_size = batch_size, shuffle=True)

    print("Number of steps per epoch: ", len(unlearning_loader))

    unlearning_teacher.eval() if unlearning_teacher is not None else None
    full_trained_teacher.eval()
    optimizer = optimizer
    if optimizer == 'adamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr = lr)
    else:
        # if optimizer is not a valid string, then assuming it as a function to return optimizer
        optimizer = optimizer#(model.parameters())

    set_seed(seed)

    for epoch in range(epochs):
        loss = unlearning_step(model = model, unlearning_teacher= unlearning_teacher, 
                        full_trained_teacher=full_trained_teacher, unlearn_data_loader=unlearning_loader, 
                        optimizer=optimizer, device=device, KL_temperature=KL_temperature, type=type)
        print("Epoch {} Unlearning Loss {}".format(epoch+1, loss))
  
    total_time = time.time() - tic
    
    return total_time

class UnLearningData(Dataset):
    def __init__(self, forget_data, retain_data, type=""):
        super().__init__()
        self.forget_data = forget_data
        self.retain_data = retain_data
        self.forget_len = len(forget_data)
        self.retain_len = len(retain_data)
        self.type = type

    def __len__(self):
        return self.retain_len + self.forget_len
    
    def __getitem__(self, index):
        if(index < self.forget_len):
            x = self.forget_data[index][0] if self.type == "cnn" else self.forget_data[index]['input_values']
            y = 1
            return x,y
        else:
            x = self.retain_data[index - self.forget_len][0] if self.type == "cnn" else self.retain_data[index - self.forget_len]['input_values']
            y = 0
            return x,y

def srl(model, retain_dataset, forget_dataset, bs, device, num_epochs=1, lr=5e-5, seed=0, type=""):

    print("Starting Successive Random Labels")

    dataset = srl_dataset(forget_dataset, retain_dataset, model.config.num_labels)
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)

    set_seed(seed)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # total_steps = len(dataloader) * num_epochs
    # warmup_steps = int(0.1 * total_steps)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    tic = time.time()
    for epoch in range(num_epochs):
        running_loss = 0
        total_samples = 0

        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for x, y in tqdm(dataloader):
            
            x = x.to(device)
            y = y.to(device)

            # Classification Loss
            outputs = model(x)
            classification_loss = criterion(outputs if type == "cnn" else outputs.logits, y)

            optimizer.zero_grad()
            classification_loss.backward()
            optimizer.step()
            # scheduler.step()

            running_loss += classification_loss.item() * x.size(0)
            total_samples += x.size(0)

        average_epoch_loss = running_loss / total_samples
        print(f"Epoch [{epoch+1}/{num_epochs}] - Total Loss: {running_loss:.4f}, Average Loss: {average_epoch_loss:.4f}")


    total_time = time.time() - tic

    return total_time

class srl_dataset(Dataset):
    def __init__(self, forget_set, retain_set, n_classes):
        super().__init__()
        self.forget_set = forget_set
        self.retain_set = retain_set
        self.forget_len = len(forget_set)
        self.retain_len = len(retain_set)
        self.n_classes = n_classes

    def __len__(self):
        return self.retain_len + self.forget_len
    
    def __getitem__(self, index):
        if(index < self.forget_len):



            x = self.forget_set[index]['input_values']
            original_label = self.forget_set[index]['labels']

            y = np.random.randint(0, self.n_classes)
            while y == original_label:
                y = np.random.randint(0, self.n_classes)
            y = torch.tensor(y)
            return x,y
        else:
            x = self.retain_set[index - self.forget_len]['input_values']
            y = torch.tensor(self.retain_set[index - self.forget_len]['labels'])

            return x,y

def saliency_map_generation(model, retain_dataloader, forget_dataloader, device, save_path, treshold = 0.5, num_epochs=1, lr=5e-5, seed=0, type=""):

    print("Starting Saliency Map Generation")

    gradients = {}

    set_seed(seed)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    model.eval()

    for name, param in model.named_parameters():
        gradients[name] = 0

    for batch in tqdm(forget_dataloader):
        if type == "cnn":
            x, y = batch
        else:
            x = batch['input_values']
            y = batch['labels']
        
        x = x.to(device)
        y = y.to(device)

        # compute output
        output_clean = model(x)
        loss = - criterion(output_clean if type == "cnn" else output_clean.logits, y)

        optimizer.zero_grad()
        loss.backward()

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    gradients[name] += param.grad.data

    with torch.no_grad():
        for name in gradients:
            gradients[name] = torch.abs_(torch.Tensor(gradients[name]).to('cpu'))

    sorted_dict_positions = {}
    hard_dict = {}

    print("Generating Saliency Map")
    all_elements = - torch.cat([tensor.flatten() for tensor in gradients.values()])

    treshold_index = int(len(all_elements) * treshold)

    positions = torch.argsort(all_elements)
    ranks = torch.argsort(positions)

    start_index = 0
    for key, tensor in tqdm(gradients.items()):
        num_elements = tensor.numel()
        # tensor_positions = positions[start_index: start_index + num_elements]
        tensor_ranks = ranks[start_index : start_index + num_elements]

        sorted_positions = tensor_ranks.reshape(tensor.shape)
        sorted_dict_positions[key] = sorted_positions

        # Set the corresponding elements to 1
        treshold_tensor = torch.zeros_like(tensor_ranks)
        treshold_tensor[tensor_ranks < treshold_index] = 1
        treshold_tensor = treshold_tensor.reshape(tensor.shape)
        hard_dict[key] = treshold_tensor
        start_index += num_elements
        
    torch.save(hard_dict, save_path)
    
    print("Saliency Map Generation Completed")
    