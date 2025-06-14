from tqdm import tqdm
import numpy as np
from PIL import Image
from datetime import datetime
import os
import argparse
import random
import torch
import torch.nn.functional as F
import os
from torch import nn, optim
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from progan_modules import Generator, Discriminator


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=(1 - decay))


def imagefolder_loader(path):
    def loader(transform):
        data = datasets.ImageFolder(path, transform=transform)
        data_loader = DataLoader(data, shuffle=True, batch_size=batch_size, num_workers=2)
        return data_loader
    return loader


def sample_data(dataloader, image_size=4):
    transform = transforms.Compose([
        transforms.Resize(image_size+int(image_size*0.2)+1),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    loader = dataloader(transform)

    return loader


def train(generator, discriminator, init_step, loader, total_iter=600000, start_iter=0, is_alpha_done=False, alpha_override=None):
    step = init_step # can be 1 = 8, 2 = 16, 3 = 32, 4 = 64, 5 = 128, 6 = 128
    data_loader = sample_data(loader, 4 * 2 ** step)
    dataset = iter(data_loader)

    #total_iter = 600000
    n_stage = 9
    total_iter_remain = total_iter - (total_iter // n_stage) * (step - 1)

    pbar = tqdm(range(total_iter_remain), dynamic_ncols=True, desc=f"Step {step} | Alpha {0:.3f}")

    disc_loss_val = 0
    gen_loss_val = 0
    grad_loss_val = 0

    
    date_time = datetime.now()
    post_fix = '%s_%s_%d_%d.txt'%(trial_name, date_time.date(), date_time.hour, date_time.minute)
    log_folder = 'trial_%s_%s_%d_%d'%(trial_name, date_time.date(), date_time.hour, date_time.minute)
    
    os.mkdir(log_folder)
    os.mkdir(log_folder+'/checkpoint')
    os.mkdir(log_folder+'/sample')

    config_file_name = os.path.join(log_folder, 'train_config_'+post_fix)
    config_file = open(config_file_name, 'w')
    config_file.write(str(args))
    config_file.close()

    log_file_name = os.path.join(log_folder, 'train_log_'+post_fix)
    log_file = open(log_file_name, 'w')
    log_file.write('g,d,nll,onehot\n')
    log_file.close()

    from shutil import copy
    copy('train.py', log_folder+'/train_%s.py'%post_fix)
    copy('progan_modules.py', log_folder+'/model_%s.py'%post_fix)

    alpha = 1.0 if is_alpha_done else 0.0
    #one = torch.FloatTensor([1]).to(device)
    one = torch.tensor(1, dtype=torch.float).to(device)
    mone = one * -1
    iteration = 0

    base_alpha = 1.0 if is_alpha_done else (alpha_override or 0.0)
    alpha = base_alpha

    for i in pbar:
        discriminator.zero_grad()

        if not is_alpha_done:
            fade_rate = 2.0 / (total_iter // n_stage)
            if alpha_override is not None and step == init_step:
                alpha = min(1.0, base_alpha + fade_rate * iteration)
            else:
                alpha = min(1.0, fade_rate * iteration)

        if iteration > total_iter // n_stage:
            alpha = 1.0 if is_alpha_done else 0.0
            iteration = 0
            step += 1

            if step > n_stage - 1:
                alpha = 1
                step = n_stage - 1
            data_loader = sample_data(loader, 4 * 2 ** step)
            dataset = iter(data_loader)

        try:
            real_image, label = next(dataset)

        except (OSError, StopIteration):
            dataset = iter(data_loader)
            real_image, label = next(dataset)

        iteration += 1

        ### 1. train Discriminator
        b_size = real_image.size(0)
        real_image = real_image.to(device)
        label = label.to(device)
        real_predict = discriminator(
            real_image, step=step, alpha=alpha)
        real_predict = real_predict.mean() \
            - 0.001 * (real_predict ** 2).mean()
        real_predict.backward(mone)

        # sample input data: vector for Generator
        gen_z = torch.randn(b_size, input_code_size).to(device)

        fake_image = generator(gen_z, step=step, alpha=alpha)
        fake_predict = discriminator(
            fake_image.detach(), step=step, alpha=alpha)
        fake_predict = fake_predict.mean()
        fake_predict.backward(one)

        ### gradient penalty for D
        eps = torch.rand(b_size, 1, 1, 1).to(device)
        x_hat = eps * real_image.data + (1 - eps) * fake_image.detach().data
        x_hat.requires_grad = True
        hat_predict = discriminator(x_hat, step=step, alpha=alpha)
        grad_x_hat = grad(
            outputs=hat_predict.sum(), inputs=x_hat, create_graph=True)[0]
        grad_penalty = ((grad_x_hat.view(grad_x_hat.size(0), -1)
                         .norm(2, dim=1) - 1)**2).mean()
        grad_penalty = 10 * grad_penalty
        grad_penalty.backward()
        grad_loss_val += grad_penalty.item()
        disc_loss_val += (real_predict - fake_predict).item()

        d_optimizer.step()

        ### 2. train Generator
        if (i + 1) % n_critic == 0:
            generator.zero_grad()
            discriminator.zero_grad()
            
            predict = discriminator(fake_image, step=step, alpha=alpha)

            loss = -predict.mean()
            gen_loss_val += loss.item()


            loss.backward()
            g_optimizer.step()
            accumulate(g_running, generator)

        if (i + 1) % 1000 == 0 or i == 0:
            with torch.no_grad():
                fake_images = g_running(torch.randn(24, input_code_size).to(device), step=step, alpha=alpha).data.cpu()

                real_images = next(iter(data_loader))[0][:4].cpu()

                combined_images = torch.cat([real_images, fake_images], dim=0)

                utils.save_image(
                    combined_images,
                    f'{log_folder}/sample/compare_{str(i + 1).zfill(6)}.png',
                    nrow=4,          
                    normalize=True,
                    scale_each=True  
                )

        if (i+1) % 5000 == 0 or i==0:
            try:
                torch.save(g_running.state_dict(), f'{log_folder}/checkpoint/{str(i + 1).zfill(6)}_g.model')
                torch.save(discriminator.state_dict(), f'{log_folder}/checkpoint/{str(i + 1).zfill(6)}_d.model')
                torch.save(g_optimizer.state_dict(), os.path.join(log_folder, 'checkpoint', f'{str(i + 1).zfill(6)}_g_optim.pth'))
                torch.save(d_optimizer.state_dict(), os.path.join(log_folder, 'checkpoint', f'{str(i + 1).zfill(6)}_d_optim.pth'))
            except:
                pass

        if (i+1) % 20 == 0 or i == 0:
            avg_gen_loss = gen_loss_val / max(1, ((i+1)//n_critic))
            avg_disc_loss = disc_loss_val / max(1, (i+1))
            avg_grad_loss = grad_loss_val / max(1, (i+1))
            pbar.set_description(
                f"Step {step} | Alpha {alpha:.3f} | G_loss {avg_gen_loss:.4f} | D_loss {avg_disc_loss:.4f} | Grad_penalty {avg_grad_loss:.4f}"
            )
        if (i+1)%500 == 0:
            log_file = open(log_file_name, 'a+')
            new_line = "%.5f,%.5f\n"%(gen_loss_val/(500//n_critic), disc_loss_val/500)
            log_file.write(new_line)
            log_file.close()

            disc_loss_val = 0
            gen_loss_val = 0
            grad_loss_val = 0
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Progressive GAN, during training, the model will learn to generate  images from a low resolution, then progressively getting high resolution ')

    parser.add_argument('--start_iter', type=int, default=0, help='Iterasi awal dari training')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint directory (default: None, train from scratch)')
    parser.add_argument('--path', type=str,default="/content/merged_dataset/Acne", help='path of specified dataset, should be a folder that has one or many sub image folders inside')
    parser.add_argument('--trial_name', type=str, default="test1", help='a brief description of the training trial')
    parser.add_argument('--gpu_id', type=int, default=0, help='0 is the first gpu, 1 is the second gpu, etc.')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default is 1e-3, usually dont need to change it, you can try make it bigger, such as 2e-3')
    parser.add_argument('--z_dim', type=int, default=128, help='the initial latent vector\'s dimension, can be smaller such as 64, if the dataset is not diverse')
    parser.add_argument('--channel', type=int, default=128, help='determines how big the model is, smaller value means faster training, but less capacity of the model')
    parser.add_argument('--batch_size', type=int, default=4, help='how many images to train together at one iteration')
    parser.add_argument('--n_critic', type=int, default=1, help='train Dhow many times while train G 1 time')
    parser.add_argument('--init_step', type=int, default=1, help='start from what resolution, 1 means 8x8 resolution, 2 means 16x16 resolution, ..., 6 means 256x256 resolution')
    parser.add_argument('--total_iter', type=int, default=300000, help='how many iterations to train in total, the value is in assumption that init step is 1')
    parser.add_argument('--pixel_norm', default=False, action="store_true", help='a normalization method inside the model, you can try use it or not depends on the dataset')
    parser.add_argument('--tanh', default=False, action="store_true", help='an output non-linearity on the output of Generator, you can try use it or not depends on the dataset')
    parser.add_argument('--isAlphaDone', default=False, action="store_true", help='jika diset, langsung pakai alpha=1 dan skip fade-in')
    parser.add_argument('--alpha', type=float, default=None, help='Starting alpha (0-1). Jika >1, akan dibagi 1000 (contoh: 276 → 0.276).')
    args = parser.parse_args()

    alpha_override = None
    if args.alpha is not None:
        if args.alpha > 1:
            alpha_override = args.alpha / 1000.0
        else:
            alpha_override = 0

    trial_name = args.trial_name
    device = torch.device("cuda:%d"%(args.gpu_id))
    input_code_size = args.z_dim
    batch_size = args.batch_size
    n_critic = args.n_critic

    generator = Generator(in_channel=args.channel, input_code_dim=input_code_size, pixel_norm=args.pixel_norm, tanh=args.tanh).to(device)
    discriminator = Discriminator(feat_dim=args.channel).to(device)
    g_running = Generator(in_channel=args.channel, input_code_dim=input_code_size, pixel_norm=args.pixel_norm, tanh=args.tanh).to(device)
    
    ## you can directly load a pretrained model here
    if args.checkpoint:
        generator_path = os.path.join(args.checkpoint, "g.model")
        discriminator_path = os.path.join(args.checkpoint, "d.model")
        optimizer_g_path = os.path.join(args.checkpoint, "g_optim.pth")
        optimizer_d_path = os.path.join(args.checkpoint, "d_optim.pth")

        if os.path.exists(generator_path) and os.path.exists(discriminator_path):
            print(f"Loading checkpoints from {args.checkpoint}...")
            generator.load_state_dict(torch.load(generator_path))
            g_running.load_state_dict(torch.load(generator_path))
            discriminator.load_state_dict(torch.load(discriminator_path))
        else:
            print(f"Warning: Checkpoint not found at {args.checkpoint}. Training from scratch!")

        g_optimizer = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.0, 0.99))
        d_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.0, 0.99))

        if os.path.exists(optimizer_g_path) and os.path.exists(optimizer_d_path):
            g_optimizer.load_state_dict(torch.load(optimizer_g_path))
            d_optimizer.load_state_dict(torch.load(optimizer_d_path))
            print("Optimizers loaded successfully!")
        else:
            print("Warning: Optimizer checkpoint not found. Using new optimizers!")
    else:
        print("No checkpoint provided, training from scratch.")
        g_optimizer = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.0, 0.99))
        d_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.0, 0.99))

    # apply weight accumulation regardless of checkpoint
    accumulate(g_running, generator, 0)

    # load dataset and start training
    loader = imagefolder_loader(args.path)
    train(generator, discriminator, args.init_step, loader, args.total_iter, args.start_iter, args.isAlphaDone, alpha_override)
