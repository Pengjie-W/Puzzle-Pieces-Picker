import torch
from .P3 import P3
from .backbone import build_backbone
from .transformer import build_transformer

def build_model(args):
    backbone = build_backbone(args)
    transformer = build_transformer(args)
    model = P3(backbone, transformer, args.num_classes)

    device = torch.device(f'cuda:{args.device}'if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
    
    return model