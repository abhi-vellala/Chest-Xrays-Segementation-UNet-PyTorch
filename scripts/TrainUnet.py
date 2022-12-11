from tqdm import tqdm
import torch
import numpy as np
from scripts.evaluate import EvaluateImageSegmentation

def train_unet(dataloaders, model, device, optimizer):
    
    train_loss_list = []
    train_dice = []
    train_accuracy = []
    train_iou = []
    train_hausdorf = []
    train_loss_dict = {}

    model.train()
    train_loss = 0.0
    load_loop = tqdm(enumerate(dataloaders["train"]), total=len(dataloaders["train"]))
    for batch_idx, (images, masks) in load_loop:
        batch_size = images.size(0)

        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        pred_masks = model(images)
        softmax = torch.nn.functional.log_softmax(pred_masks, dim=1)
        loss = torch.nn.CrossEntropyLoss()
        loss = loss(softmax, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * batch_size
        train_loss = train_loss / len(load_loop)
        train_loss_list.append(train_loss)
        
        
        gt = np.array(masks.cpu(), dtype=int)
        pred = np.array(pred_masks.cpu().detach().permute(0, 1, 2, 3)[:,-1, :, :], dtype=int)

        metric_evaluate = EvaluateImageSegmentation(gt, pred)
        train_dice.append(metric_evaluate.dice())
        train_accuracy.append(metric_evaluate.accuracy())
        train_iou.append(metric_evaluate.IoU())
        train_hausdorf.append(metric_evaluate.hausdorff_distance('euclidean'))
        
        


        load_loop.set_postfix(loss=loss.item())
    
    train_loss_dict['loss'] = train_loss_list 
    train_loss_dict['dice'] = train_dice
    train_loss_dict['accuracy'] = train_accuracy
    train_loss_dict['iou'] = train_iou
    train_loss_dict['hausdorf'] = train_hausdorf
    
        

    return train_loss_dict

def evaluate_unet(dataloaders, model, device):

    model.eval()
    validation_loss = 0.0
    validation_loss_list = []
    validation_dice = []
    validation_accuracy = []
    validation_iou = []
    validation_hausdorf = []
    validation_loss_dict = {}
    for images, masks in dataloaders["valid"]:
        batch_size = images.size(0)

        images = images.to(device)
        masks = masks.to(device)

        with torch.no_grad():
            pred_masks = model(images)
            softmax = torch.nn.functional.log_softmax(pred_masks, dim=1)
            validation_loss += torch.nn.functional.nll_loss(softmax, masks).item() * batch_size
            validation_loss_list.append(validation_loss)
        
        
        gt = np.array(masks.cpu(), dtype=int)
        pred = np.array(pred_masks.cpu().detach().permute(0, 1, 2, 3)[:,-1, :, :], dtype=int)

        metric_evaluate = EvaluateImageSegmentation(gt, pred)
        validation_dice.append(metric_evaluate.dice())
        validation_accuracy.append(metric_evaluate.accuracy())
        validation_iou.append(metric_evaluate.IoU())
        validation_hausdorf.append(metric_evaluate.hausdorff_distance('euclidean'))
        
    
    validation_loss_dict['loss'] = validation_loss_list 
    validation_loss_dict['dice'] = validation_dice
    validation_loss_dict['accuracy'] = validation_accuracy
    validation_loss_dict['iou'] = validation_iou
    validation_loss_dict['hausdorf'] = validation_hausdorf
    
        

    return validation_loss_dict
            