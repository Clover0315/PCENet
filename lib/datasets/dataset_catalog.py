from lib.config import cfg


class DatasetCatalog(object):
    dataset_attrs = {
        'AiCrowdTrain': {
            'id': 'aicrowd',
            'data_root': 'data/aicrowd/train/images',
            'ann_file': 'data/aicrowd/train/annotation.json',
            'split': 'train'
        },
        'AiCrowdVal': {
            'id': 'aicrowd',
            'data_root': 'data/aicrowd/val/val/images/',
            'ann_file': 'data/aicrowd/val/val/annotation.json',
            'split': 'test'
        },
        'WhuTrain': {
            'id': 'whu',
            'data_root': 'data/whumix-train/image',
            'ann_file': 'data/whumix-train/train.json',
            'split': 'train'
        },
        'WhuVal': {
            'id': 'whu',
            'data_root': 'data/whumix-val/image',
            'ann_file': 'data/whumix-val/val.json',
            'split': 'test'
        }
    }

    @staticmethod
    def get(name):
        attrs = DatasetCatalog.dataset_attrs[name]
        return attrs.copy()
