from argparse import ArgumentParser
import torch.optim as optim #import SGD,RMSprop
import torch
from Dataloader import get_data_loaders
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from tqdm import tqdm
from FushionNet import FushionNet
import torch.nn.functional as F

SAVE_PATH="./weights/"

def run(train_batch_size, val_batch_size, epochs, lr, momentum, log_interval):
    train_loader, val_loader = get_data_loaders(train_batch_size, val_batch_size)
    model = FushionNet()
    #model=torch.load(SAVE_PATH+"350-0.908.pth")


    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum,weight_decay=2e-6,nesterov=False)
    #optimizer = optim.Adamax(model.parameters(),lr,(0.9,0.999),1e-8,1e-6)
    scheduler =optim.lr_scheduler.MultiStepLR(optimizer,[50,100,150,200,250,300],0.1)
    trainer = create_supervised_trainer(model, optimizer, F.nll_loss, device=device)
    evaluator = create_supervised_evaluator(model,
                                            metrics={'accuracy': Accuracy(),
                                                     'nll': Loss(F.nll_loss)},
                                            device=device)

    desc = "ITERATION - loss: {:.2f}"
    pbar = tqdm(
        initial=0, leave=False, total=len(train_loader),
        desc=desc.format(0)
    )

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1

        if iter % log_interval == 0:
            pbar.desc = desc.format(engine.state.output)
            pbar.update(log_interval)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        pbar.refresh()
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['nll']
        current_lr=optimizer.param_groups[0]['lr']
        tqdm.write(
            "Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f} Current lr: {:.6f}"
                .format(engine.state.epoch, avg_accuracy, avg_nll,current_lr)
        )
        scheduler.step()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_nll = metrics['nll']
        tqdm.write(
            "Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
                .format(engine.state.epoch, avg_accuracy, avg_nll))
        pbar.n = pbar.last_print_n = 0
        if (engine.state.epoch%10==0):
            torch.save(model, SAVE_PATH+str(engine.state.epoch)+"-"+str(avg_accuracy)+".pth")

    trainer.run(train_loader, max_epochs=epochs)
    pbar.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--val_batch_size', type=int, default=64,
                        help='input batch size for validation (default: 1000)')
    parser.add_argument('--epochs', type=int, default=360,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.2,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='how many batches to wait before logging training status')

    args = parser.parse_args()

run(args.batch_size, args.val_batch_size, args.epochs, args.lr, args.momentum, args.log_interval)

