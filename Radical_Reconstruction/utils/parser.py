import argparse
from get_dataset.get_src_vocab import get_src_vocab

class DefaultParser(object):

    def __init__(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('--train_dataset', type=str, default='OBS_train')
        parser.add_argument('--test_dataset', type=str, default='OBS_test')
        parser.add_argument('--batch_aug', action='store_true') # True

        # Sequence Construction
        parser.add_argument('--max_num_text_ins', type=int, default=60) # Inherited from SPTS

        # Model Configuration
        parser.add_argument('--backbone', type=str, default='resnet50')
        parser.add_argument('--position_embedding', type=str, default='sine')
        parser.add_argument('--tfm_hidden_dim', type=int, default=256)
        parser.add_argument('--tfm_dropout', type=float, default=0.1)
        parser.add_argument('--tfm_nheads', type=int, default=8)
        parser.add_argument('--tfm_dim_feedforward', type=int, default=1024)
        parser.add_argument('--tfm_enc_layers', type=int, default=6)
        parser.add_argument('--tfm_dec_layers', type=int, default=6)
        parser.add_argument('--tfm_pre_norm', action='store_true',default=True) # True

        # Training Parameters
        parser.add_argument('--lr', type=float, default=0.0005)
        parser.add_argument('--min_lr', type=float, default=0.000001)
        parser.add_argument('--warmup_min_lr', type=float, default=0.0001)
        parser.add_argument('--lr_backbone_ratio', type=float, default=0.1) 
        parser.add_argument('--finetune', action='store_true')
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--batch_size', type=int, default=256)
        parser.add_argument('--num_workers', type=int, default=16)
        parser.add_argument('--freeze_bn', action='store_true',default=True) # True
        parser.add_argument('--eos_loss_coef', type=float, default=0.01)
        parser.add_argument('--epochs', type=int, default=100)
        parser.add_argument('--warmup_epochs', type=int, default=5)
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--eval', action='store_true',default=False)
        parser.add_argument('--test', action='store_true',default=False)
        parser.add_argument('--resume', type=str, default='')
        parser.add_argument('--output_folder', type=str,default='./output/Deciphering_dataset/OBS_train/')
        parser.add_argument('--print_freq', type=int, default=10)
        parser.add_argument('--source', type=str, default='../Dataset_Generation/source.json')
        parser.add_argument('--checkpoint_freq', type=int, default=100)
        parser.add_argument('--max_norm', type=float, default=0.1)
        parser.add_argument('--device', type=int, default=0)
        parser.add_argument('--retrain', action='store_true',default=False)

        # Beam Search Parameters, only False
        parser.add_argument('--beam_search', action='store_true',default=False)
        parser.add_argument('--beam_size', type=int, default=100)
        parser.add_argument('--beam_search_max_length', type=int, default=24)
        parser.add_argument('--use_length_penalty', action='store_true',default=False)
        parser.add_argument('--length_penalty', type=float, default=0.7)


        # Distributed Parameters
        parser.add_argument('--local_rank', type=int, default=0)
        parser.add_argument('--world_size', type=int, default=1)
        parser.add_argument('--rank', type=int, default=0)

        self.parser = parser

    def add_argument(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)
    
    def parse_args(self):
        args = self.parser.parse_args()
        src_vocab=get_src_vocab(args)
        args.recog_pad_index = src_vocab['<pad>']
        args.eos_index=src_vocab['<eos>']
        args.sos_index=src_vocab['<sos>']
        args.padding_index=src_vocab['<pad>']
        args.num_classes=len(src_vocab)
        return args
        