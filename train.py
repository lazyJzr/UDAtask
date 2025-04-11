import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from TransQAA import TransAxials, Classifier
from U_net import Unet
from dataloader import read_directory
from GaussianDiffusion import GaussianDiffusion
from diffusion_scheduler_factory import create_gaussian_diffusion
from utils import parse_args, compute_accuracy

args = parse_args()


print('Training model...')

model = Unet(
        dim = 32,
        channels=24,
        dim_mults=(1, 2, 4,)
    )
model = model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_diff)

f0_C = Classifier()
f0_C = f0_C.cuda()
criterion = nn.CrossEntropyLoss()
optimizer_class = optim.Adam(f0_C.parameters())
for param_group in optimizer_class.param_groups:
    param_group['lr'] = args.lr_class

feature_model = TransAxials()

diffusion = create_gaussian_diffusion(
        normalize_input=True,
        schedule_name=args.schedule_name,
        min_noise_level=1e-3,
        steps=args.num_steps,
        kappa=1.0,
        etas_end=0.95,
        schedule_kwargs={"power": 2},  
        weighted_mse=False,
        timestep_respacing=10, 
        scale_factor=args.scale_factor,
        latent_flag=True,
    )

for t in range(args.num_epoch):
    print(f'The {t + 1}th epoch: ')
    for inputs, labels in target_train_dataloader:
        dataset = feature_model(inputs)
        dataset = dataset.detach().cuda()
        batch_size = inputs.shape[0]

        loss_reservse = diffusion.diffusion_loss_fn(model, dataset, args.num_steps)
        optimizer.zero_grad()
        loss_reservse.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()

    print('Target reversal training complete ！！')

    for batch_idx, (data_s, target_s) in enumerate(source_train_dataloader):
        dataset = feature_model(data_s)
        labels = target_s.cuda()
        data_s = data_s.cuda()
        optimizer_class.zero_grad()
        
        f0_C.train(dataset.detach().cuda(), labels.detach(), criterion, optimizer_class, -1)
        
    print('Classifier pre-training complete ！！')
    
    f0_C.eval()

    FD_list = []
    label_list = []
    for step in range(1, args.num_steps+1):
        print(f'The {step}th diffusion step: ')
        for batch_idx, (batch_s, batch_t) in enumerate(zip(source_train_dataloader, target_train_dataloader)):
            data_s, target_s = batch_s
            data_s = data_s.detach()

            with torch.no_grad():
                dataset = feature_model(data_s)

            data_s, target_s = data_s.cuda(), target_s.cuda()
            data_t, _ = batch_t
            with torch.no_grad():
                dataset_t = feature_model(data_t)
            data_t = data_t.cuda()
            dataset = dataset.detach().cuda()
            dataset_t = dataset_t.detach().cuda()
            
            if args.domain_strategy == 'residual':
                loss_diff = diffusion.diffusion_loss_fn(model, dataset, step, y=dataset_t-dataset)
                FDi = diffusion.p_sample_loop_e(model, dataset_t.shape, step, dataset_t-dataset)[-1].detach()
            elif args.domain_strategy == 'noresidual':
                loss_diff = diffusion.diffusion_loss_fn(model, dataset, step)
                FDi = diffusion.p_sample_loop(model, dataset_t.shape, step)[-1].detach()
                
            else:
                raise ValueError(f"Unknown residual mode: {args.residual}. Expected 'residual' or 'noresidual'.")

            loss_class = criterion(f0_C(FDi), target_s)
            
            loss = loss_diff + loss_class
            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            
            if step % args.save_interval == 0:
                FD_list.append(FDi)
                label_list.append(target_s)
        
        if step % args.save_interval == 0:
            all_FDi = torch.cat(FD_list, dim=0)
            all_labels = torch.cat(label_list, dim=0)
            f0_C.train(all_FDi.detach(), all_labels.detach(), criterion, optimizer_class, -1)
    print('Training completed ！！！')




with torch.no_grad():
    features = feature_model(test_features)
test_features = test_features.cuda()
f0_C = f0_C.eval()
_, teac = compute_accuracy(f0_C, features.cuda(), test_labels)
print('Test set accuracy: ', teac.item(), '%')














