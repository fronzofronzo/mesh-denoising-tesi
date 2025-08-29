import optuna
import torch
import torch.optim as Toptim
import torch.nn.functional as F
import h5py
import numpy as np
from parsers import getParser
from datautils import MatrixDataset
from train import splitData
from GCNModel import DGCNN

k_opt = getParser()

def objective(trial):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- HYPERPARAMETER SEARCH SPACE ---- 
    lr = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [32,64,128,256])
    data_number = 644771
    val_batch_size = int(data_number * 0.3)
    num_train_batch = int((data_number/batch_size) * 0.7)

    data_path_file = k_opt.data_path_file
    data_path = h5py.File(data_path_file, 'r')
    data_path = np.array(data_path["data_path"])
    train_path, val_path = splitData(data_path, val_batch_size)

    train_dataset = MatrixDataset(k_opt, train_path, k_opt.num_neighbors, is_train=True, batch_size=batch_size)
    train_data_loader = train_dataset.getDataloader()

    val_dataset = MatrixDataset(k_opt, val_path, k_opt.num_neighbors, is_train=False, batch_size=batch_size)
    val_data_loader = val_dataset.getDataloader()

    # ---- MODEL ----
    dgcnn = DGCNN(8, 18, 1024, 0.5).to(device)
    optimizer = Toptim.Adam(dgcnn.parameters(), lr=lr, betas=(0.9, 0.999))

    cos_target = torch.tensor(np.ones((batch_size)))
    cos_target = cos_target.type(torch.FloatTensor).cuda()

    # ---- TRAIN ----
    for epoch in range(4):
        dgcnn.train()
        for i_train, data in enumerate(train_data_loader):
            inputs, gt_res, gt_norm, center_norm = data
            inputs = inputs.type(torch.FloatTensor)
            inputs = inputs.permute(0, 2, 1)
            gt_norm = gt_norm.type(torch.FloatTensor)

            inputs = inputs.cuda()
            gt_norm = gt_norm.cuda()

            optimizer.zero_grad()
            output = dgcnn(inputs)

            cos_loss = F.cosine_embedding_loss(output, gt_norm, cos_target)
            value_loss = F.mse_loss(output, gt_norm)
            loss = value_loss
            loss.backward()
            optimizer.step()

            print("Epoch: %d, || Batch: %d/%d, || cos loss: %.7f, || value loss: %.7f" % \
                (epoch, i_train + 1, num_train_batch, cos_loss.data.item(), value_loss.data.item()))

    # ---- VALIDATION ----
    dgcnn.eval()
    val_losses = []
    with torch.no_grad():
        for inputs, gt_res, gt_norm, center_norm in val_data_loader:
            inputs = inputs.type(torch.FloatTensor)
            inputs = inputs.permute(0, 2, 1)
            gt_norm = gt_norm.type(torch.FloatTensor)

            inputs = inputs.cuda()
            gt_norm = gt_norm.cuda()

            output = dgcnn(inputs)
            cos_loss = F.cosine_embedding_loss(output, gt_norm, cos_target)
            value_loss = F.mse_loss(output, gt_norm)
            loss = value_loss
            val_losses.append(loss.item())

    
    return np.mean(val_losses)

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)

    print("Best trial:")
    trial = study.best_trial
    print(f" Loss: {trial.value}")
    print(" Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
