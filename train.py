from opt import opt
import os
import torch

from evaluate import evaluate_model

def save_network_LR(name,
                model,
                optimizer,
                scheduler,
                loss,
                epoch = "last",
                models_dir = "models"):

    checkpoint = {
        'epoch': epoch,
        'model': model.cpu().state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'loss' : loss,
        'name': name
    }

    save_filename = 'net_%s_%s.pth'% (name, epoch)
    os.makedirs(models_dir, exist_ok=True)
    save_path = os.path.join(models_dir, save_filename)

    torch.save(checkpoint, save_path)
    if opt.usecpu == False and torch.cuda.is_available():
        model.cuda()

def save_network(name,
                model,
                epoch = "last",
                models_dir = "models"):

    checkpoint = {
        'epoch': epoch,
        'model': model.cpu().state_dict(),
        'name': name
    }

    save_filename = 'net_%s_%s.pth'% (name, epoch)
    os.makedirs(models_dir, exist_ok=True)
    save_path = os.path.join(models_dir, save_filename)

    torch.save(checkpoint, save_path)
    if opt.usecpu == False and torch.cuda.is_available():
        model.cuda()

def load_network(name,
                model,
                epoch = "last",
                models_dir = "models"):

    load_filename = 'net_%s_%s.pth'% (name, epoch)
    load_path = os.path.join(models_dir, load_filename)

    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model'])

    if opt.usecpu == False and torch.cuda.is_available():
        model.cuda()

def load_network_LR(name,
                model,
                optimizer,
                scheduler,
                epoch = "last",
                models_dir = "models"):

    load_filename = 'net_%s_%s.pth'% (name, epoch)
    load_path = os.path.join(models_dir, load_filename)

    checkpoint = torch.load(load_path)

    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])

    if opt.usecpu == False and torch.cuda.is_available():
        model.cuda()

def train_model(app):

    start_epoch = 1
    if (opt.resume_epoch != -1):
        load_network("MGN",
                    app.model,
                    opt.resume_epoch)
        start_epoch = opt.resume_epoch

    phases = ['train', 'validate']
    loaders = {}
    loaders['train'] = app.train_loader
    loaders['validate'] = app.train_val_loader
    hist_batch = {}
    hist_batch['train'] = 0
    hist_batch['validate'] = 0
    for epoch in range(start_epoch, opt.epoch + 1):

        # Each epoch has a training and validation phase
        for phase in phases:

            running_loss = {}
            running_loss['total'] = 0
            running_loss['triplet'] = 0
            running_loss['ce'] = 0

            if phase == 'train':
                app.model.train(True)  # Set model to training mode
            else:
                app.model.train(False)  # Set model to evaluate mode
            epoch_batch = 0
            for _, (inputs, labels) in enumerate(loaders[phase]):

                if opt.usecpu == False and torch.cuda.is_available():
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                # zero the parameter gradients
                app.optimizer.zero_grad()

                # forward
                if phase == 'validate':
                    with torch.no_grad():
                        outputs = app.model(inputs)
                else:
                    outputs = app.model(inputs)

                # compute inference loss
                loss_sum, triplet_loss, ce_loss = app.loss(outputs, labels)

                # Log loss per mini batch
                app.writer.add_scalar('total_loss/batch/%s' % phase,
                                    loss_sum.data.cpu().numpy(), hist_batch[phase])
                running_loss['total'] += loss_sum.data.cpu().numpy()
                app.writer.add_scalar('triplet_loss/batch/%s' % phase,
                                    triplet_loss.data.cpu().numpy(), hist_batch[phase])
                running_loss['triplet'] += triplet_loss.data.cpu().numpy()
                app.writer.add_scalar('ce_loss/batch/%s' % phase,
                                    ce_loss.data.cpu().numpy(), hist_batch[phase])
                running_loss['ce'] += ce_loss.data.cpu().numpy()
                app.writer.flush()

                # back propagate the computed gradient
                if phase == 'train':
                    loss_sum.backward()
                    app.optimizer.step()
                    app.scheduler.step()

                epoch_batch = epoch_batch + 1
                hist_batch[phase] = hist_batch[phase] + 1

            # Log loss per epoch
            avg_total_loss = running_loss['total'] /  epoch_batch
            app.writer.add_scalar('total_loss/epoch/%s' % phase,
                                avg_total_loss, epoch)
            avg_triplet_loss = running_loss['triplet'] /  len(loaders[phase])
            app.writer.add_scalar('triplet_loss/epoch/%s' % phase,
                                avg_triplet_loss, epoch)
            avg_ce_loss = running_loss['ce'] /  epoch_batch
            app.writer.add_scalar('ce_loss/epoch/%s' % phase,
                                avg_ce_loss, epoch)
            app.writer.flush()

            print('Epoch %d Phase %s Total loss:%.2f  Triplet_Loss:%.2f  CE_Loss:%.2f' % (
                epoch,
                phase,
                avg_total_loss,
                avg_triplet_loss,
                avg_ce_loss))

        if epoch % opt.savefreq == 0:
            save_network(
                "MGN",
                app.model,
                epoch)

            if epoch in opt.lr_scheduler:
                save_network_LR(
                    "MGN_LR",
                    app.model,
                    app.optimizer,
                    app.scheduler,
                    app.loss,
                    epoch)

        if epoch % opt.evalfreq == 0 or epoch == 1:
            evaluate_model(app, epoch)

    save_network(
            "MGN",
            app.model,
            "last")
    save_network_LR(
            "MGN_LR",
            app.model,
            app.optimizer,
            app.scheduler,
            app.loss,
            "last")
    app.model.cpu()
