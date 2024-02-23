import time
import datetime

import numpy as np
import torch
import tqdm
from torch.nn import DataParallel

from lib.utils import net_utils
from lib.utils.snake import snake_whu_utils, snake_coco_utils
from lib.utils.snake.matcher import HungarianMatcher


class Trainer(object):
    def __init__(self, network):
        network = network.cuda()
        network = DataParallel(network)
        self.network = network

    def reduce_loss_stats(self, loss_stats):
        reduced_losses = {k: torch.mean(v) for k, v in loss_stats.items()}
        return reduced_losses

    def to_cuda(self, batch):
        for k in batch:
            if k == 'meta':
                continue
            if isinstance(batch[k], tuple):
                batch[k] = [b.cuda() for b in batch[k]]
            else:
                batch[k] = batch[k].cuda()
        return batch

    def train(self, epoch, data_loader, optimizer, recorder):
        max_iter = len(data_loader)
        self.network.train()
        end = time.time()
        for iteration, batch in enumerate(data_loader):
            data_time = time.time() - end
            iteration = iteration + 1
            recorder.step += 1

            # batch = self.to_cuda(batch)
            output, loss, loss_stats, image_stats = self.network(batch)

            # training stage: loss; optimizer; scheduler
            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.network.parameters(), 40)
            optimizer.step()

            # data recording stage: loss_stats, time, image_stats
            loss_stats = self.reduce_loss_stats(loss_stats)
            recorder.update_loss_stats(loss_stats)

            batch_time = time.time() - end
            end = time.time()
            recorder.batch_time.update(batch_time)
            recorder.data_time.update(data_time)

            if iteration % 20 == 0 or iteration == (max_iter - 1):
                # print training state
                eta_seconds = recorder.batch_time.global_avg * (max_iter - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                lr = optimizer.param_groups[0]['lr']
                memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0

                training_state = '  '.join(['eta: {}', '{}', 'lr: {:.6f}', 'max_mem: {:.0f}'])
                training_state = training_state.format(eta_string, str(recorder), lr, memory)
                print(training_state)

                # record loss_stats and image_dict
                recorder.update_image_stats(image_stats)
                recorder.record('train')

    def val(self, epoch, data_loader, evaluator=None, recorder=None, reduce_by_km=False, eval_tp=False):
        self.network.eval()
        torch.cuda.empty_cache()
        val_loss_stats = {}
        data_size = len(data_loader)

        if eval_tp:
            TP, TN, FP, FN = 0, 0, 0, 0
            matcher = HungarianMatcher()
        for batch in tqdm.tqdm(data_loader):
            for k in batch:
                if k != 'meta':
                    batch[k] = batch[k].cuda()

            with torch.no_grad():
                output, loss, loss_stats, image_stats = self.network(batch)

                # TODO
                if reduce_by_km:
                    '''km 算法匹配py和'''
                    py = output['py']
                    py = py[-1] if isinstance(py, list) else py
                    py = py.detach().cpu().numpy()
                    gt = batch['meta']['gt_py']
                    pys_valid = []

                    for i in range(len(gt)):
                        py_pred = py[i]
                        py_gt = gt[i].detach().cpu().numpy()[0]
                        indexes_cost = snake_whu_utils.cal_munkres(py_pred, py_gt, matrix_type='L2')

                        py_valid = py_pred[indexes_cost, :]
                        py_valid = snake_coco_utils.uniformsample(py_valid, 128)
                        pys_valid.append(py_valid)

                    output['py'].append(torch.from_numpy(np.array(pys_valid)))

                if evaluator is not None:
                    evaluator.evaluate(output, batch)

                # EVAL todo
                if eval_tp:
                    outputs = {'pred_prob': net_utils.sigmoid(output['pred_ep_tp']),
                               'pred_poly': output['init_ep_tp']}
                    targets = {'gt_num': output['gt_pys_num'], 'gt_py': output['gt_pys']}
                    device = outputs['pred_poly'].device
                    if outputs['pred_poly'].size(0) and targets['gt_py'].size(0):
                        # 匈牙利匹配
                        indices = matcher(outputs, targets, 1)
                        pred_prob = outputs['pred_prob']
                        # 分类矩阵
                        gt_arr = []
                        for indice in indices:
                            gt = torch.zeros_like(pred_prob[0]).to(device)
                            gt[indice[0]] = 1
                            gt_arr.append(gt)
                        gt_tensor = torch.stack(gt_arr)
                        pred_tensor = torch.where(pred_prob >= 0.6, 1, 0)
                        add_matrix = pred_tensor + gt_tensor
                        diff_matrix = pred_tensor - gt_tensor
                        TP += int(torch.where(add_matrix == 2, 1, 0).sum().detach().cpu())
                        TN += int(torch.where(add_matrix == 0, 1, 0).sum().detach().cpu())
                        FN += int(torch.where(diff_matrix == -1, 1, 0).sum().detach().cpu())
                        FP += int(torch.where(diff_matrix == 1, 1, 0).sum().detach().cpu())

            loss_stats = self.reduce_loss_stats(loss_stats)
            for k, v in loss_stats.items():
                val_loss_stats.setdefault(k, 0)
                val_loss_stats[k] += v

        loss_state = []
        for k in val_loss_stats.keys():
            val_loss_stats[k] /= data_size
            loss_state.append('{}: {:.4f}'.format(k, val_loss_stats[k]))
        print(loss_state)

        if evaluator is not None:
            result = evaluator.summarize()
            val_loss_stats.update(result)

        if recorder:
            recorder.record('val', epoch, val_loss_stats, image_stats)

        if eval_tp:
            print(
                '''TP:{},TN:{},FP:{},FN:{},precision:{:.2f}%,recall:{:.2f}%,accuracy:{:.2f}%'''.format(
                    TP, TN, FP, FN, (TP / (TP + FP)) * 100, (TP / (TP + FN)) * 100, (TP + TN) / (TP+TN+FN+FP) * 100
                )
            )









