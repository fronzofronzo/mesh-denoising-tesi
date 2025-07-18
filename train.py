import os
import json
import csv
from datetime import datetime

import torch
import torch.optim as Toptim
import torch.utils.data
import torch.nn.functional as F

import h5py
import numpy as np
import random
import matplotlib.pyplot as plt

from GCNModel import DGCNN
from parsers import getParser
from datautils import MatrixDataset

from tensorboardX import SummaryWriter

k_opt = getParser()
k_epoch = k_opt.num_epoch

k_loss_writer = SummaryWriter('runs/losses')

if not os.path.exists(k_opt.val_res_path):
    os.makedirs(k_opt.val_res_path)

if not os.path.exists(k_opt.ckpt_path):
    os.makedirs(k_opt.ckpt_path)

log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def setup_logging():
    """
    Function to setup logging files.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_log_file = os.path.join(log_dir, f"training_log_{timestamp}.json")

    csv_log_file = os.path.join(log_dir, f"training_metrics_{timestamp}.csv")

    txt_log_file = os.path.join(log_dir, f"training_output_{timestamp}.txt")

    with open(csv_log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_cos_loss', 'train_value_loss', 'val_loss',
                        'val_cos_loss', 'val_value_loss', 'timestamp'])
    
    with open(json_log_file, 'w') as f:
        json.dump({"training_start": timestamp, 'epochs': []}, f, indent=2)

    return json_log_file, csv_log_file, txt_log_file

def log_epoch_data(epoch, train_metrics, val_metrics, json_log_file, csv_log_file, text_log_file):
    """
    Function to save epoch datas on a log file.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(json_log_file, 'r') as f:
        data = json.load(f)
    
    epoch_data = {
        'epoch': epoch,
        'timestamp': timestamp,
        'train_loss': float(train_metrics['loss']),
        'train_cos_loss': float(train_metrics['cos_loss']),
        'train_value_loss': float(train_metrics['value_loss']),
        'val_loss': float(train_metrics['loss']),
        'val_cos_loss': float(val_metrics['cos_loss']),
        'val_value_loss': float(val_metrics['value_loss']),
    }

    data['epochs'].append(epoch_data)

    with open(json_log_file, 'w') as f:
        json.dump(data, f, indent=2)

    with open(csv_log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, train_metrics['loss'], train_metrics['cos_loss'], train_metrics['value_loss'],
        val_metrics['loss'], val_metrics['cos_loss'], val_metrics['value_loss'], timestamp])
    
    with open(text_log_file, 'a') as f:
        f.write(f"[{timestamp}] Epoch {epoch}:\n")
        f.write(f" Train - Loss: {train_metrics['loss']:.7f}, Cos: {train_metrics['cos_loss']:.7f}, Value: {train_metrics['value_loss']:.7f}\n")
        f.write(f" Val - Loss: {val_metrics['loss']:.7f}, Cos: {val_metrics['cos_loss']:.7f}, Value: {val_metrics['value_loss']:.7f}\n")
        f.write("-" * 80 + "\n")

def load_previous_log(json_log_file):
    """
    Upload precedent files, if they exist. 
    """
    if os.path.exists(json_log_file):
        with open(json_log_file, 'r') as f:
            data = json.load(f)
            return data.get('epochs', [])
    return []
        
def plot_losses_realtime(history_data, save_path='training_progress.png'):
    """
    Function that creates loss graphs real time.
    """ 

    if not history_data:
        return

    epochs = [d['epoch'] for d in history_data]
    train_cos = [d['train_cos_loss'] for d in history_data]
    val_cos = [d['val_cos_loss'] for d in history_data]
    train_value = [d['train_value_loss'] for d in history_data]
    val_value = [d['val_value_loss'] for d in history_data]

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(epochs, train_cos, '-o', label='Training cosine loss')
    plt.plot(epochs, val_cos, '-o', label='Validation cosine loss')
    plt.xlabel('Epoch')
    plt.ylabel('Cosine Loss')
    plt.title('Cosine Loss Progress')
    plt.legend()
    plt.grid()
    
    plt.subplot(1,2,2)
    plt.plot(epochs, train_value, '-o', label='Training value loss')
    plt.plot(epochs, val_value, '-o', label='Validation value loss')
    plt.xlabel('Epoch')
    plt.ylabel('Value Loss')
    plt.title('Value Loss Progress')
    plt.legend()
    plt.grid()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def splitData(data_path, num_val_batch):
    num_data = data_path.shape[0]
    print(num_val_batch * k_opt.batch_size)
    num_val_data = num_val_batch * k_opt.batch_size
    num_train_data = num_data - num_val_data

    val_index = random.sample(range(0, num_data), num_val_data)
    train_index = list(set(range(0, num_data)) - set(val_index))

    train_path = data_path[train_index]
    val_path = data_path[val_index]

    val_index = np.array(val_index)
    np.save(k_opt.val_res_path + "val_index.npy", val_index)

    return train_path, val_path

def reSplitData(data_path):
    num_data = data_path.shape[0]
    val_index = np.load(k_opt.val_res_path + "val_index.npy")
    val_index = list(val_index)
    train_index = list(set(range(0, num_data)) - set(val_index))

    train_path = data_path[train_index]
    val_path = data_path[val_index]

    return train_path, val_path

def train():
    json_log_file, csv_log_file, text_log_file = setup_logging()
    
    data_path_file = k_opt.data_path_file
    data_path = h5py.File(data_path_file, 'r')
    data_path = np.array(data_path["data_path"])

    if k_opt.current_model != "":
        train_path, val_path = reSplitData(data_path)
    else:
        train_path, val_path = splitData(data_path, k_opt.num_val_batch)
    num_train_batch = int(train_path.shape[0] / k_opt.batch_size)

    # initialize Dataloader
    train_dataset = MatrixDataset(k_opt, train_path, k_opt.num_neighbors, is_train=True)
    train_data_loader = train_dataset.getDataloader()

    val_dataset = MatrixDataset(k_opt, val_path, k_opt.num_neighbors, is_train=False)
    val_data_loader = val_dataset.getDataloader()

    # initialize Network structure etc.
    current_epoch = 0
    dgcnn = DGCNN(8, 17, 1024, 0.5)
    # dgcnn = torch.nn.DataParallel(dgcnn)
    if k_opt.current_model != "":
        dgcnn.load_state_dict(torch.load(k_opt.current_model))
        print("Load ", k_opt.current_model, " Success!")
        current_epoch = int(k_opt.current_model.split('/')[-1].split('_')[0]) + 1
    optimizer = Toptim.Adam(dgcnn.parameters(), lr=k_opt.learning_rate, betas=(0.9, 0.999))
    dgcnn.cuda()

    cos_target = torch.tensor(np.ones((k_opt.batch_size)))
    cos_target = cos_target.type(torch.FloatTensor).cuda()
    weight_alpha = 0.
    weight_beta = 1.

    last_val_cos_loss = 999.
    last_val_value_loss = 999.
   
    previous_history = load_previous_log(json_log_file)
    
    with open(text_log_file, 'a') as f:
        f.write(f'=== TRAINING STARTED AT {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} === \n')
        f.write(f'Starting from epoch: {current_epoch}\n')
        f.write(f'Total epochs: {k_epoch}\n')
        f.write(f'Batch size: {k_opt.batch_size}\n')
        f.write(f'Learning rate: {k_opt.learning_rate}\n')
        f.write('=' * 80 + '\n')
   
    for epoch in range(current_epoch, k_epoch):
        train_loss = []
        train_cos_loss = []
        train_value_loss = []
        for i_train, data in enumerate(train_data_loader, 0):
            inputs, gt_res, gt_norm, center_norm = data
            inputs = inputs.type(torch.FloatTensor)
            inputs = inputs.permute(0, 2, 1)
            gt_norm = gt_norm.type(torch.FloatTensor)

            inputs = inputs.cuda()
            gt_norm = gt_norm.cuda()

            optimizer.zero_grad()
            dgcnn = dgcnn.train()

            output = dgcnn(inputs)
            
            print(output.shape)
            print(gt_norm.shape)

            cos_loss = F.cosine_embedding_loss(output, gt_norm, cos_target)
            value_loss = F.mse_loss(output, gt_norm)

            if(i_train % 100 == 0):
                k_loss_writer.add_scalar('cos_loss', cos_loss, global_step=epoch * num_train_batch + i_train + 1)
                k_loss_writer.add_scalar('value_loss', value_loss, global_step=epoch * num_train_batch + i_train + 1)

            loss = weight_alpha * cos_loss + weight_beta * value_loss
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.data.item())
            train_cos_loss.append(cos_loss.data.item())
            train_value_loss.append(value_loss.data.item())

            print("Epoch: %d, || Batch: %d/%d, || cos loss: %.7f, || value loss: %.7f, || val cos loss: %.7f || val value loss: %.7f" % \
                (epoch, i_train + 1, num_train_batch, cos_loss.data.item(), value_loss.data.item(), last_val_cos_loss, last_val_value_loss))
        
        train_metrics = {
            'loss': np.mean(train_loss),
            'cos_loss': np.mean(train_cos_loss),
            'value_loss': np.mean(train_value_loss)
        }
        
        #______Validation______
        torch.save(dgcnn.state_dict(), k_opt.ckpt_path + str(epoch) + "_model.t7")
        val_cos_loss = []
        val_value_loss = []
        val_loss = []
        dgcnn.eval()
        for i_val, data in enumerate(val_data_loader, 0):
            inputs, gt_res, gt_norm, center_norm = data
            inputs = inputs.type(torch.FloatTensor)
            inputs = inputs.permute(0, 2, 1)
            gt_norm = gt_norm.type(torch.FloatTensor)

            inputs = inputs.cuda()
            gt_norm = gt_norm.cuda()

            output = dgcnn(inputs)
            
            print(output.shape)
            print(gt_norm.shape)
            print(cos_target.shape)

            cos_loss = F.cosine_embedding_loss(output, gt_norm, cos_target)
            value_loss = F.mse_loss(output, gt_norm)

            loss = weight_alpha * cos_loss + weight_beta * value_loss

            val_loss.append(loss.data.item())
            val_cos_loss.append(cos_loss.data.item())
            val_value_loss.append(value_loss.data.item())

            print("Epoch: %d, || Val Batch: %d/%d, || cos loss: %.7f, || value loss: %.7f" % \
                (epoch, i_val + 1, k_opt.num_val_batch, cos_loss.data.item(), value_loss.data.item()))
        
        val_metrics = {
            'loss': np.mean(val_loss),
            'cos_loss': np.mean(val_cos_loss),
            'value_loss': np.mean(val_value_loss)
        }
        
        last_val_cos_loss = val_metrics['cos_loss']
        last_val_value_loss = val_metrics['value_loss']
        
        log_epoch_data(epoch, train_metrics, val_metrics, json_log_file, csv_log_file, text_log_file)
        
        current_history = load_previous_log(json_log_file)
        plot_losses_realtime(current_history, os.path.join(log_dir, 'train_progress.png'))

        k_loss_writer.add_scalar('val_cos_loss', last_val_cos_loss, global_step=epoch + 1)
        k_loss_writer.add_scalar('val_value_loss', last_val_value_loss, global_step=epoch + 1)
        
        print(f'Epoch {epoch} completed and logged')
        
    # Log finale
    with open(text_log_file, 'a') as f:
        f.write(f"=== TRAINING COMPLETED AT {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    
    print(f"Training completed! Check logs in: {log_dir}")
    print(f"- JSON log: {json_log_file}")
    print(f"- CSV log: {csv_log_file}")
    print(f"- Text log: {text_log_file}")

if __name__ == '__main__':
    train()
