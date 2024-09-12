import torch
from utils import Metric, MetricGroup
import utils.args as args

def validation(dataloader, model, criterion):
    metrics = MetricGroup([Metric('valid_avg_loss', 0.0), Metric('valid_avg_acc1', 0.0), Metric('valid_avg_acc5', 0.0)])
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['HE'].cuda()
            targets = batch['label'].cuda()
            logits = model(inputs)
            loss = criterion(logits, targets)

            # measure accuracy and record loss
            _, top5_preds = logits.topk(5, 1, True, True)
            top5_preds = top5_preds.t()
            top1_preds = top5_preds[0]
            top1_acc = torch.sum(top1_preds == targets)/targets.size(0)
            top5_acc = torch.sum(torch.sum(top5_preds == targets, dim=0, dtype=torch.bool))/targets.size(0)
            metrics.step([loss.detach(), top1_acc, top5_acc])
    return metrics.avg


def train(dataloader, model, criterion, optimizer, regularization=None):
    metrics = MetricGroup([Metric('train_avg_loss', 0.0), Metric('train_avg_acc', 0.0)]) # trace loss/acc of every batch
    metrics_log = MetricGroup([Metric('train_log_loss', 0.0), Metric('train_log_acc', 0.0)]) # trace loss/acc for each log_freq

    model.train()
    len_epoch = len(dataloader)
    for step, batch in enumerate(dataloader):
        inputs = batch['HE'].cuda()
        targets = batch['label'].cuda()
        logits = model(inputs)
        loss = criterion(logits, targets)
        if regularization is not None:
            reg = regularization(model)
            loss = loss + args.lamb * reg

        # update model
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # gradient clipping to reproduce
        optimizer.step()
        _, preds = logits.max(1)

        # statistics
        metrics.step([loss.detach(), torch.sum(preds == targets.data)/targets.size(0)])
        metrics_log.step([loss.detach(), torch.sum(preds == targets.data)/targets.size(0)])

        # Log during an epoch
        if (step+1) % args.log_freq == 0:
            print(f"\tStep {step+1}/{len_epoch}: {metrics_log}")
            metrics_log.reset()
    return metrics.avg
    
