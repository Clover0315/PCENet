from lib import get_task_name

TRAIN_TASK_NAME = get_task_name()
if TRAIN_TASK_NAME == "whu":
    from .config_whu import cfg
elif TRAIN_TASK_NAME == 'aicrowd':
    from .config_aicrowd import cfg
elif TRAIN_TASK_NAME == 'ep_tp':
    from .config_ep_tp import cfg


