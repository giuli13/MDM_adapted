from torch.utils.data import DataLoader
from data_loaders.tensors import collate as all_collate
from data_loaders.tensors import t2m_collate, t2m2p_collate, t2m_mixed_collate, truebones_mixed_collate

def get_dataset_class(name):
    if name == "amass":
        from .amass import AMASS
        return AMASS
    elif name == "uestc":
        from .a2m.uestc import UESTC
        return UESTC
    elif name == "humanact12":
        from .a2m.humanact12poses import HumanAct12Poses
        return HumanAct12Poses
    elif name == "humanml":
        from data_loaders.humanml.data.dataset import HumanML3D
        return HumanML3D
    elif name == "kit":
        from data_loaders.humanml.data.dataset import KIT
        return KIT
    elif name=="truebones":
        from data_loaders.truebones.data.dataset import TruebonesMixedJoints
        return TruebonesMixedJoints
    elif name=="humanml_mat":
        from data_loaders.humanml.data.dataset import HumanML3DMixedJoints
        return HumanML3DMixedJoints
    else:
        raise ValueError(f'Unsupported dataset name [{name}]')

def get_collate_fn(name, hml_mode='train'):
    if hml_mode == 'gt':
        from data_loaders.humanml.data.dataset import collate_fn as t2m_eval_collate
        return t2m_eval_collate
    if name in ["humanml", "kit"]:
        return t2m_collate
    if name in ["truebones"]:
        return truebones_mixed_collate
    if name in ["humanml_mat"]:
        return t2m_mixed_collate
    else:
        return all_collate


def get_dataset(name, num_frames, split='train', hml_mode='train', debug=False, temporal_window=31, hml=False, t5_name='t5-base', balanced=False, tpos_not_normalized=False, objects_subset="all"):
    DATA = get_dataset_class(name)
    if name == "humanml_mat":
        dataset = DATA(split=split, num_frames=num_frames, mode=hml_mode, debug=debug)
    elif name in ["humanml", "kit", "truebones"]:
        dataset = DATA(split=split, num_frames=num_frames, mode=hml_mode, temporal_window=temporal_window, hml=hml, t5_name=t5_name, balanced=balanced, tpos_not_normalized=tpos_not_normalized, objects_subset=objects_subset)
    else:
        dataset = DATA(split=split, num_frames=num_frames, mode="train")
    return dataset


def get_dataset_loader(name, batch_size, num_frames, split='train', hml_mode='train', debug=False, temporal_window=31, hml=False, t5_name='t5-base', balanced=False, tpos_not_normalized=False, objects_subset="all"):
    if hml_mode == 'gt':
        name='humanml'
    dataset = get_dataset(name, num_frames, split, hml_mode, debug, temporal_window, hml, t5_name, balanced, tpos_not_normalized, objects_subset)
    collate = get_collate_fn(name, hml_mode)
    sampler = None
    if name == 'truebones' and balanced: #create batch sampler
        from data_loaders.truebones.data.dataset import TruebonesSampler
        sampler = TruebonesSampler(dataset)
    
    loader = DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, shuffle=True if sampler is None else False,
        num_workers=8, drop_last=True, collate_fn=collate
    )
    return loader