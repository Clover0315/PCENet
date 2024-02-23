from lib import set_task_name

set_task_name('ep_tp')
import warnings

warnings.filterwarnings('ignore')

from lib.config import cfg
from lib.networks import make_network
from lib.train import make_trainer, make_optimizer, make_lr_scheduler, make_recorder, set_lr_scheduler
from lib.datasets import make_data_loader
from lib.utils.net_utils import load_model, save_model, load_network
from lib.utils.snake import snake_gcn_whu_utils
from lib.evaluators import make_evaluator


def train(cfg, network):
    snake_gcn_whu_utils.seed_everything()

    trainer = make_trainer(cfg, network)
    optimizer = make_optimizer(cfg, network)
    scheduler = make_lr_scheduler(cfg, optimizer)
    recorder = make_recorder(cfg)
    evaluator = make_evaluator(cfg)

    load_network(network.backbone.dla, cfg.dla_model_dir, epoch=-1, strict=True)  # 加载预训练的dla34模型
    begin_epoch = load_model(network, optimizer, scheduler, recorder, cfg.model_dir, resume=cfg.resume)

    set_lr_scheduler(cfg, scheduler)

    train_loader = make_data_loader(cfg, is_train=True)
    val_loader = make_data_loader(cfg, is_train=False)

    for epoch in range(begin_epoch, cfg.train.epoch):
        recorder.epoch = epoch
        trainer.train(epoch, train_loader, optimizer, recorder)
        scheduler.step()

        if (epoch + 1) % cfg.save_ep == 0:
            save_model(network, optimizer, scheduler, recorder, epoch, cfg.model_dir)

        if (epoch + 1) % 1 == 0:
            trainer.val(epoch, val_loader, evaluator, recorder)

    return network


def test(cfg, network):
    trainer = make_trainer(cfg, network)
    cfg.test_km = True
    val_loader = make_data_loader(cfg, is_train=False)
    evaluator = make_evaluator(cfg)

    epoch = load_network(network, cfg.model_dir)
    trainer.val(epoch, val_loader, evaluator)


def main(is_train=False):
    network = make_network(cfg)
    # network.unfreeze_dla()  # 解冻dla
    network.freeze_dla()  # 冻结dla

    if is_train:
        cfg.use_gt_det = True
        train(cfg, network)
    else:
        cfg.use_gt_det = True
        test(cfg, network)


if __name__ == "__main__":
    main(True)
