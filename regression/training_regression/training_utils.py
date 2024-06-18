import torch

from tqdm import tqdm

# Training function.
def train(model,num_classes, trainloader, optimizer, criterion, device):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        features = data["features"]
        labels = data["target"]
        # taking care of the batch dimension. The zeroth-dimension has to be of the batch
        if num_classes == 1:
          labels = labels.unsqueeze(1)
        # transfer all the vectors to the same devivce
        features = features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # Forward pass.
        outputs = model(features)
        # Calculate the loss.
        
        loss = criterion(outputs, labels)
        # normalize by the maximum value
        train_running_loss += loss.item()/torch.max(torch.abs(labels)).item()
        # Backpropagation
        loss.backward()
        # Update the weights.
        optimizer.step()
    
    # average loss for the epoch
    epoch_loss = train_running_loss / counter
    return epoch_loss

# Training function.
def train_MVE(model,num_classes, trainloader, optimizer, criterion, device):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        features = data["features"]
        labels = data["target"]
        # taking care of the batch dimension. The zeroth-dimension has to be of the batch
        if num_classes == 1:
          labels = labels.unsqueeze(1)
        # transfer all the vectors to the same devivce
        features = features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # Forward pass.
        outputs = model(features)
        # Calculate the loss.
        
        loss = criterion(outputs[:,0].unsqueeze(1), labels,torch.square(outputs[:,1].unsqueeze(1)))
        # normalize by the maximum value
        train_running_loss += loss.item()/torch.max(torch.abs(labels)).item()
        # Backpropagation
        loss.backward()
        # Update the weights.
        optimizer.step()
    
    # average loss for the epoch
    epoch_loss = train_running_loss / counter
    return epoch_loss


# Validation function.
def validate(model,num_classes, testloader, criterion, device):
    model.eval()
    print('Validation')
    mse_running_loss = 0.0
    counter = 0
    counter_samples = 0
    linf_running_loss = 0.0

    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            features = data["features"]
            labels = data["target"]
            # couting the number of samples in the validation set. Required for CV style training
            counter_samples += features.size()[0]
            features = features.to(device)
            labels = labels.to(device)
            # Forward pass.
            outputs = model(features)
            
            # takes care of the batch size
            if num_classes == 1:
                labels = labels.unsqueeze(1)

            # Calculate the loss.
            loss = criterion(outputs, labels)
            linf_running_loss += torch.sum(torch.div(torch.abs(outputs-labels),torch.abs(labels)))
            mse_running_loss += torch.mean(torch.div(torch.square(outputs-labels),torch.square(labels))).item() 
       
    # Loss and accuracy for the complete epoch.
    return linf_running_loss.item()/counter_samples,mse_running_loss/counter

# Validation function.
def validate_MVE(model,num_classes, testloader, criterion, device):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    mse_running_loss = 0.0
    counter = 0
    counter_samples = 0
    variance = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            features = data["features"]
            labels = data["target"]
            # couting the number of samples in the validation set. Required for CV style training
            counter_samples += features.size()[0]
            features = features.to(device)
            labels = labels.to(device)
            # Forward pass.
            outputs = model(features)
            
            # Calculate the loss.
            # pointwise relative error (usually larger than MSE, might not converge that fast)
            valid_running_loss += torch.sum(torch.div(torch.abs(outputs[:,0]-labels),torch.abs(labels)))
            variance += torch.sum(torch.square(outputs[:,1]))
            mse_running_loss += torch.mean(torch.div(torch.square(outputs[:,0]-labels),torch.square(labels))).item() 
    # pointwise relative error per sample
    loss_per_sample = valid_running_loss / counter_samples
    return loss_per_sample.item(),variance/counter_samples, mse_running_loss/counter
