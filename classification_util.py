from argparse import Namespace
from collections import OrderedDict
import os
import pickle
os.environ['CUDA_LAUNCH_BLOCKING']='1'
from lifelines.utils import concordance_index
import numpy as np
from sksurv.metrics import concordance_index_censored
from models.model_porpoise import CMAF
from torch.nn import MSELoss, CrossEntropyLoss
from utils.utils import *
import torch

from datasets.dataset_generic import save_splits

def train(datasets: tuple, cur: int, model: torch.nn.Module, args: Namespace):
    """
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split = datasets
    save_splits(datasets, ['train', 'val'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))

    print('\nInit loss function...', end=' ')

    loss_fn = CrossEntropyLoss()
    cls_loss = CrossEntropyLoss()

    if args.reg_type == 'omic':
        reg_fn = l1_reg_omic
    elif args.reg_type == 'pathomic':
        reg_fn = l1_reg_modules
    else:
        reg_fn = None

    if hasattr(model, "relocate"):
        model.relocate()
    else:
        model = model.to(torch.device('cuda'))
    print('Done!')

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')

    schedular = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5, last_epoch=-1)

    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing=args.testing,
                                    weighted=args.weighted_sample, mode=args.mode, batch_size=args.batch_size)
    val_loader = get_split_loader(val_split, testing=args.testing, mode=args.mode, batch_size=args.batch_size)
    print('Done!')

    for epoch in range(args.max_epochs):
        train_loop_classification(epoch, model, train_loader, optimizer, args.n_classes, args.mode, writer, loss_fn, cls_loss, reg_fn,
                            args.lambda_reg, args.gc)
        stop = validate_classification(cur, epoch, model, val_loader, args.n_classes, args.mode, writer, loss_fn, cls_loss, reg_fn,
                                 args.lambda_reg, args.results_dir)
        schedular.step()
    torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))
    model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))

    results_val_dict, val_cindex = summary_classification(model, val_loader, args.n_classes, args.mode)

    print('Val c-Index: {:.4f}'.format(val_cindex))
    writer.close()
    return results_val_dict, val_cindex

def train_loop_classification(epoch, model, loader, optimizer, n_classes, mode, writer, loss_fn, cls_loss, reg_fn,
                              lambda_reg, gc):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    train_loss, train_loss_stage = 0., 0.
    pos = 0
    all_pred = []
    all_censorships = []
    all_event_times = []
    for batch_idx, (data_WSI, data_omic, y_disc, event_time, censor, stage) in enumerate(loader):
        T = 1 / (batch_idx + 1)
        data_WSI, data_omic = data_WSI.to(device), data_omic.to(device)
        y_disc = y_disc.to(device)
        stage = stage.to(device)
        if mode == 'pathomic':
            predict, modality_pred, modality_label = model(x_path=data_WSI, x_omic=data_omic)
            modality_label = modality_label.to(device)
        else:
            h, predict = model(x_path=data_WSI, x_omic=data_omic)

        pred_stage = predict.argmax()
        if pred_stage.item() == stage.item():
            pos += 1
        loss_stage = loss_fn(predict, stage)
        if mode == 'pathomic':
            loss_modality = cls_loss(modality_pred, modality_label)
        else:
            loss_modality = 0
        loss = loss_stage + T * loss_modality
        loss_value = loss.item()

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg
        all_pred.append(predict.detach().cpu().numpy())
        all_censorships.append(censor.numpy())
        all_event_times.append(event_time.numpy())
        train_loss_stage += loss_stage.item()
        train_loss += loss_value + loss_reg

        # backward pass
        loss = loss / gc + loss_reg
        loss.backward()

        if (batch_idx + 1) % gc == 0:
            optimizer.step()
            optimizer.zero_grad()

    # calculate loss and error for epoch
    train_loss_stage /= len(loader)
    train_loss /= len(loader)
    stage_acc = pos / len(loader)

    print('Epoch: {}, train_loss_surv: {:.4f}, train_loss: {:.4f}, train_acc: {:.4f}'.format(epoch, train_loss_stage, train_loss, stage_acc))

    if writer:
        writer.add_scalar('train/loss_surv', train_loss_stage, epoch)
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/acc', stage_acc, epoch)


def validate_classification(cur, epoch, model, loader, n_classes, mode, writer, loss_fn, cls_loss, reg_fn,
                            lambda_reg, results_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    val_loss, val_loss_stage = 0., 0.
    pos = 0
    all_pred = []
    all_censorships = []
    all_event_times = []
    for batch_idx, (data_WSI, data_omic, y_disc, event_time, censor, stage) in enumerate(loader):
        T = 1 / (batch_idx + 1)
        data_WSI, data_omic = data_WSI.to(device), data_omic.to(device)
        y_disc = y_disc.to(device)
        event_time = event_time.to(device)
        censor = censor.to(device)
        stage = stage.to(device)
        with torch.no_grad():
            if mode == 'pathomic':
                predict, modality_pred, modality_label = model(x_path=data_WSI, x_omic=data_omic)
                modality_label = modality_label.to(device)
            else:
                h, predict = model(x_path=data_WSI, x_omic=data_omic)

        pred_stage = predict.argmax()
        if pred_stage.item() == stage.item():
            pos += 1
        loss_stage = loss_fn(predict, stage)
        if mode == 'pathomic':
            loss_modality = cls_loss(modality_pred, modality_label)
        else:
            loss_modality = 0
        loss = loss_stage + T * loss_modality
        loss_value = loss.item()

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg
        all_pred.append(predict)
        all_censorships.append(censor.detach().cpu().numpy())
        all_event_times.append(event_time.detach().cpu().numpy())
        val_loss_stage += loss_stage.item()
        val_loss += loss_value + loss_reg

        if y_disc.shape[0] == 1 and (batch_idx + 1) % 100 == 0:
            print('batch {}, loss: {:.4f}, label: {}, event_time: {:.4f}'.format(batch_idx, loss_value + loss_reg, y_disc.detach().cpu().item(), float(event_time.detach().cpu().item())))
        elif y_disc.shape[0] != 1 and (batch_idx + 1) % 5 == 0:
            print('batch {}, loss: {:.4f}, label: {}, event_time: {:.4f}'.format(batch_idx, loss_value + loss_reg, y_disc.detach().cpu()[0], float(event_time.detach().cpu()[0])))

    # calculate loss and error for epoch
    val_loss_stage /= len(loader)
    val_loss /= len(loader)
    stage_acc = pos / len(loader)

    print('Epoch: {}, val_loss_surv: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}'.format(epoch, val_loss_stage, val_loss, stage_acc))

    if writer:
        writer.add_scalar('val/loss_surv', val_loss_stage, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/acc', stage_acc, epoch)


def summary_classification(model, loader, n_classes, mode):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_loss = 0.
    pos = 0
    all_pred = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, (data_WSI, data_omic, y_disc, event_time, censor, stage) in enumerate(loader):
        data_WSI, data_omic = data_WSI.to(device), data_omic.to(device)
        slide_id = slide_ids.iloc[batch_idx]

        with torch.no_grad():
            if mode == 'pathomic':
                predict, modal_pred, modal_label = model(x_path=data_WSI, x_omic=data_omic)
            else:
                predict = model(x_omic=data_omic, x_path=data_WSI)

        pred_stage = predict.argmax()
        if pred_stage.item() == stage.item():
            pos += 1

        predict = predict.detach().cpu().tolist()
        event_time = event_time.item()
        censor = censor.item()
        # all_pred[batch_idx] = predict
        # all_censorships[batch_idx] = censor
        # all_event_times[batch_idx] = event_time
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'pred': predict, 'disc_label': y_disc.item(), 'survival': event_time, 'censorship': censor}})

    acc = pos / len(loader)

    return patient_results, acc
