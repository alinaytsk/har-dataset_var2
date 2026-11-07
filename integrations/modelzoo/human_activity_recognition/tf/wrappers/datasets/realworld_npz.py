"""Реєстрація завантажувача har-dataset (.npz) у реєстрі ST model zoo.

Повторює структуру realworld_imu.py, тільки без validation_split/test_split:
спліт зафіксований у самих файлах датасету.
"""
from common.registries.dataset_registry import DATASET_WRAPPER_REGISTRY
from human_activity_recognition.tf.src.datasets.realworld_npz import load_realworld_npz
from human_activity_recognition.tf.src.datasets.utils import prepare_kwargs_for_dataloader


@DATASET_WRAPPER_REGISTRY.register(framework='tf', dataset_name='realworld_npz',
                                   use_case="human_activity_recognition")
def get_realworld_npz(cfg):
    args = prepare_kwargs_for_dataloader(cfg)
    if args.get('gravity_rot_sup'):
        raise ValueError(
            "preprocessing.gravity_rot_sup має бути False для цього датасету: "
            "обертання по гравітації прибирає постійну складову акселерометра, "
            "якою Lying відрізняється від Stationary, і прошивка його не робить.")
    train_ds, val_ds, test_ds = load_realworld_npz(
        training_path=args['training_path'],
        class_names=args['class_names'],
        input_shape=args['input_shape'],
        batch_size=args['batch_size'],
        seed=args['seed'],
        normalization=args['normalization'],
        to_cache=args['to_cache'])
    return {'train': train_ds, 'valid': val_ds, 'test': test_ds}
