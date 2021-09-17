from torchvision import transforms, datasets, models
import torch
from torch import optim, cuda
import torch.nn as nn
import numpy as np
import pandas as pd
from timeit import default_timer as timer
# from torchsummary import summary
import matplotlib.pyplot as plt
from model import save_final_model, load_trained_resnet50
import csv
import seaborn as sb
import os



def create_loss_and_optimizer(model, lr, lr_step=7, lr_decrease=0.1):
    criterion = nn.BCEWithLogitsLoss()  # BCEWithLogitsLoss = `Sigmoid` layer + `BCELoss`
    optimizer = optim.Adam(model.parameters(), lr=lr)
    print(f'Learning rate: {lr}')

    # Read carefully about where to put lr_scheduler on https://pytorch.org/docs/stable/optim.html
    # Decay learning rate by a factor of gamma (0.1) every step_size (7) epochs
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer,
                                             step_size=lr_step,
                                             gamma=lr_decrease)

    # for p in optimizer.param_groups[0]['params']:
    #     if p.requires_grad:
    #         print(p.shape)

    return criterion, optimizer, lr_scheduler


def model_train(model, criterion, optimizer, lr_scheduler,
                train_loader, valid_loader, checkpoint_dir, log_dir,
                early_stop_epoch_limit=3, n_epochs=20, weight_decay=0.3, print_every=2):
    """Train a PyTorch Model

    Params
    --------
        model (PyTorch model): cnn to train
        criterion (PyTorch loss): objective to minimize
        optimizer (PyTorch optimizer): optimizer to compute gradients of model parameters
        train_loader (PyTorch dataloader): training dataloader to iterate through
        valid_loader (PyTorch dataloader): validation dataloader used for early stopping
        checkpoint_dir (str ending in '.pt'): file path to save the checkpoint model state dict
        early_stop_epoch_limit (int): maximum number of epochs with no improvement in validation loss
        for early stopping
        n_epochs (int): maximum number of training epochs
        print_every (int): frequency of epochs to print training stats

    Returns
    --------
        model (PyTorch model): trained cnn with best weights
        history (DataFrame): history of train and validation loss and accuracy
    """

    # Check if gpu is available
    train_on_gpu = cuda.is_available()

    # Early stopping initialization
    epochs_no_improve = 0
    best_epoch = 0
    valid_loss_min = np.Inf

    # valid_max_acc = 0
    history = []

    # Number of epochs already trained (if using loaded in model weights)
    try:
        print(f'Model has been trained for: {model.epochs} epochs.\n')
    except AttributeError:
        model.epochs = 0
        print(f'Starting Training from 0 epoch.\n')

    fieldnames = ['epoch', 'train_acc', 'train_loss', 'valid_acc', 'valid_loss']
    with open(log_dir, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

    overall_start = timer()

    # Epochs loop
    for epoch in range(n_epochs):
        train_loss = 0.0
        valid_loss = 0.0
        train_acc = 0.0
        valid_acc = 0.0

        # Set to training mode
        model.train()
        start = timer()

        # Batches loop
        for index, (data, target) in enumerate(train_loader):
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()

            # Clear gradients
            optimizer.zero_grad()
            output = model(data)
            # print(f'min output: {torch.min(output)}; max output: {torch.max(output)}')
            loss = criterion(output, target)
            # print(f'loss: {loss}')
            # print(f'loss shape: {loss.shape}')
            loss.backward()

            # weight_decay
            # 0.3 is best with lr = 3e-3 in fastai experience
            # Implement of weight decay like in https://www.fast.ai/2018/07/02/adam-weight-decay/
            for group in optimizer.param_groups:
                for param in group['params']:
                    param.data = param.data.add(- weight_decay * group['lr'], param.data)

            # Update the parameters
            optimizer.step()

            # Reduce lr every epoch for torch.optim.lr_scheduler.OneCycleLR and CyclicLR only
            # lr_scheduler.step()

            # Track train loss by multiplying average loss by number of examples in batch
            train_loss += loss.item() * data.size(0)

            # Calculate accuracy by assign all value > 0 to 1, other to 0
            output[output > 0] = 1
            output[output < 0] = 0

            # Tensor of Boolean comparing output and target, size(correct_tensor) = batch_size
            correct_tensor = torch.all(torch.eq(output, target), dim=1)
            # Total number of correct predictions, size(batch_acc) = 1
            batch_acc = \
                torch.sum(correct_tensor.type(torch.FloatTensor)).cpu().detach().numpy()
            # accuracy = batch_acc / correct_tensor.shape[0]

            train_acc += batch_acc
            ##########################################
            # Track training progress
            print(
                f'Epoch: {epoch}\t{100 * (index + 1) / len(train_loader):5.2f}% complete. '
                f'{timer() - start:6.2f} seconds elapsed in epoch.'
                f'\tLearning rate: {lr_scheduler.get_lr()[0]:.2e}',
                end='\r'
            )

        # After each batches loop (1 epoch) ends, start validating
        else:
            model.epochs += 1

            model.eval()
            # Don't need to keep track of gradients
            with torch.no_grad():
                for data, target in valid_loader:
                    if train_on_gpu:
                        data, target = data.cuda(), target.cuda()

                    output = model(data)
                    loss = criterion(output, target)

                    valid_loss += loss.item() * data.size(0)

                    #####################################
                    # Calculate validation accuracy
                    output[output <= 3] = 0
                    output[output > 3] = 1

                    # Tensor of Boolean comparing output and target, size(correct_tensor) =
                    # batch_size
                    correct_tensor = torch.all(torch.eq(output, target), dim=1)
                    # Total number of correct predictions, size(batch_acc) = 1
                    batch_acc = \
                        torch.sum(correct_tensor.type(torch.FloatTensor)).cpu().detach().numpy()
                    # accuracy = batch_acc / correct_tensor.shape[0]
                    valid_acc += batch_acc
                    #####################################

                # Calculate average losses
                train_loss = round(train_loss / len(train_loader.dataset), 4)
                valid_loss = round(valid_loss / len(valid_loader.dataset), 4)

                # Calculate average accuracy
                train_acc = round(100 * train_acc / len(train_loader.dataset), 2)
                valid_acc = round(100 * valid_acc / len(valid_loader.dataset), 2)

                history.append([train_loss, valid_loss, train_acc, valid_acc])

                # Print training and validation results
                if (epoch + 1) % print_every == 0:
                    print(
                        f'\nEpoch: {epoch} \tTraining Loss: {train_loss}'
                        f'\tValidation Loss: {valid_loss}'
                        f'\t\tTraining Accuracy: {train_acc}%'
                        f'\tValidation Accuracy: {valid_acc}%'
                        f'\n\t\tLearning rate: {lr_scheduler.get_lr()[0]:.2e}'
                        f'\tLearning rate 2: {optimizer.param_groups[0]["lr"]:.2e}'
                    )

                with open(log_dir, 'a', newline='') as csv_file:
                    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                    writer.writerow({
                        'epoch': epoch,
                        'train_acc': train_acc,
                        'train_loss': train_loss,
                        'valid_acc': valid_acc,
                        'valid_loss': valid_loss
                        })

                # Save the model if validation loss decreases
                if valid_loss < valid_loss_min:
                    print(
                        f'\t\tModel improved at epoch {epoch}: validation loss from '
                        f'{valid_loss_min:.4f} to {valid_loss:.4f}'
                    )

                    # Save model
                    # torch.save(model.state_dict(), checkpoint_dir)
                    model.optimizer = optimizer
                    save_final_model(model, checkpoint_dir)


                    # Track improvement
                    epochs_no_improve = 0
                    valid_loss_min = valid_loss
                    valid_best_acc = valid_acc
                    best_epoch = epoch

                # Otherwise increment count of epochs with no improvement
                else:
                    epochs_no_improve += 1
                    print(
                        f'\t\tModel has not improved for {epochs_no_improve} epochs.'
                        f'Best epoch is {best_epoch}'
                    )
                    # Trigger early stopping
                    if epochs_no_improve >= early_stop_epoch_limit:
                        print(
                            f'\nEarly Stopping! Total epochs: {epoch}. Best epoch: {best_epoch} '
                            f'with loss: {valid_loss_min:.4f} and acc: {valid_best_acc:.2f}%'
                        )
                        total_time = timer() - overall_start
                        print(
                            f'{total_time:.2f} total seconds elapsed. '
                            f'{total_time / (epoch + 1):.2f} seconds per epoch in average.'
                        )
                        # Load the best state dict
                        model, optimizer = load_trained_resnet50( checkpoint_dir )

                        # Attach the optimizer
                        model.optimizer = optimizer

                        # Format history
                        history = pd.DataFrame(
                            history, columns=['train_loss', 'valid_loss', 'train_acc', 'valid_acc'])

                        return model, history

        # Reduce lr every epoch for torch.optim.lr_scheduler.StepLR
        lr_scheduler.step()

    # End training after all epochs
    # Load the best state dict
    model.load_state_dict(torch.load(checkpoint_dir))

    # Attach the optimizer
    model.optimizer = optimizer
    # Record overall time and print out stats
    total_time = timer() - overall_start
    print(
        f'\nBest epoch: {best_epoch} with loss: {valid_loss_min:.4f} and acc: '
        f'{valid_best_acc:.2f}%'
    )
    print(f'{total_time:.2f} total seconds elapsed. '
          f'{total_time / n_epochs:.2f} seconds per epoch in average.')
    # Format history
    history = pd.DataFrame(history, columns=['train_loss', 'valid_loss', 'train_acc', 'valid_acc'])

    return model, history


def valid_mass_predict(model, test_loader, log_dir):

    # initialize confusion matrix
    idx_to_class = model.idx_to_class
    total_classes = len(idx_to_class)
    confusion_matrix = torch.zeros( size=( total_classes, total_classes ), dtype=torch.int )

    # Check if gpu is available
    train_on_gpu = cuda.is_available()

    # Number of epochs already trained (if using loaded in model weights)
    try:
        print(f'Model has been trained for: {model.epochs} epochs.\n')
    except AttributeError:
        model.epochs = 0
        print(f'Model has not been trained.\n')

    fieldnames = ['image_path', 'predict_base_0',
                  # 'predict_base_1', 'predict_base_2',
                  'predict_base_3',
                  'predict', 'target']
    with open(log_dir, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, delimiter=',', lineterminator='\n',
                                fieldnames=fieldnames)
        writer.writeheader()

    overall_start = timer()
    test_acc_0 = 0
    accuracy_0 = np.empty(0)
    test_acc_3 = 0
    accuracy_3 = np.empty(0)

    # Batches loop
    model.eval()
    # Don't need to keep track of gradients
    with torch.no_grad():
        for index, (image_paths, data, target) in enumerate(test_loader):
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
            output = model(data)

            # Calculate validation accuracy
            output_0 = torch.zeros_like(output)
            output_0[output < 0] = 0
            output_0[output > 0] = 1
            correct_tensor_0 = torch.all(torch.eq(output_0, target),  dim=1)
            batch_acc_0 = \
                torch.sum(correct_tensor_0.type(torch.FloatTensor)).cpu().detach().numpy()
            accuracy_0 = np.append(accuracy_0, batch_acc_0 / correct_tensor_0.shape[0])
            test_acc_0 += batch_acc_0

            # output_1 = torch.zeros_like(output)
            # output_1 -= 1
            # output[output < -1] = 0
            # output[output > 1] = 1
            #
            # output_2 = torch.zeros_like(output)
            # output_2 -= 1
            # output[output < -2] = 0
            # output[output > 2] = 1

            output_3 = torch.zeros_like(output)
            output_3 -= 1
            output_3[output < -3] = 0
            output_3[output > 3] = 1

            correct_tensor_3 = torch.all(torch.eq(output_3, target), dim=1)
            batch_acc_3 = torch.sum(correct_tensor_3.type(torch.FloatTensor)).cpu().detach().numpy()
            accuracy_3 = np.append(accuracy_3, batch_acc_3 / correct_tensor_3.shape[0])
            test_acc_3 += batch_acc_3

            # Set confusion matrix at threshold 3
            _, output_idx = torch.max(output, dim=1)
            _, target_idx = torch.max(target, dim=1)
            for idx1 ,idx2 in zip( target_idx, output_idx ):
                confusion_matrix[idx1.item(), idx2.item()] += 1

            #####################################
            with open(log_dir, 'a', newline='') as csv_file:
                writer = csv.DictWriter(csv_file, delimiter=',', lineterminator='\n',
                                        fieldnames=fieldnames)
                for i, image_path in enumerate(image_paths):
                    data_dict = {
                        'image_path': image_path,
                        'predict_base_0': correct_tensor_0[i].cpu().detach().numpy(),
                        'predict_base_3': correct_tensor_3[i].cpu().detach().numpy(),
                        'predict': output[i].cpu().detach().tolist(),
                        'target': target[i].type(torch.IntTensor).cpu().detach().tolist()
                    }
                    writer.writerow(data_dict)

            print(
                f'\tBatch {100 * (index + 1) / len(test_loader):5.2f}% '
                f'({index+1:{int(np.log10(len(test_loader)))+1}d} / {len(test_loader)}) complete '
                f'\t{timer() - overall_start:.2f} seconds '
                f'\tAccuracy 0: {100 * np.mean(accuracy_0):.2f}% '
                f'\tAccuracy 3: {100 * np.mean(accuracy_3):.2f}% ',
                end='\r'
            )

    # Calculate average accuracy
    final_test_acc_0 = round(100 * test_acc_0 / len(test_loader.dataset), 2)
    final_test_acc_3 = round(100 * test_acc_3 / len(test_loader.dataset), 2)

    total_time = timer() - overall_start
    # Print training and validation results
    print(f'\n\tTime run: {total_time:.2f} seconds.'
          f'\nFinal_test_acc_0: {final_test_acc_0}%'
          f'\nFinal_test_acc_3: {final_test_acc_3}%')

    # Save confusion matrix
    class_names = list(idx_to_class.values())
    df_cfm = pd.DataFrame( confusion_matrix.cpu().numpy(), index=class_names,  columns= class_names)
    plt.figure(figsize=(10,7))
    cfm_plot = sb.heatmap(df_cfm, annot=True, xticklabels   =class_names, yticklabels=class_names, fmt="d")
    cfm_plot.figure.savefig(os.path.join( os.path.split(log_dir)[0] , "confusion_matrix.png" ))