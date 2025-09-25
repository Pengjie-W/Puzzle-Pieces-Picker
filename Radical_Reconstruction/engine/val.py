import os
import json
import torch
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm
from utils.nested_tensor import NestedTensor
from get_dataset.get_src_vocab import get_src_vocab



@torch.no_grad()
def validate(model, dataloader, epoch, args):
    model.eval()
    device = torch.device(f'cuda:{args.device}')
    file_path = args.hanzi
    with open(file_path, 'r', encoding='utf-8') as file:
        radical_dataset = json.load(file)
    file_path = args.key_to_hanzi
    with open(file_path, 'r', encoding='utf-8') as file:
        key_to_hanzi = json.load(file)
    src_vocab = get_src_vocab(args)
    results = {}
    total=0
    right=0
    bleu=0
    zhongwendata=[]
    test_bar=tqdm(dataloader)
    for inputs, input_seqs,output_seqs,path,zhongwen in test_bar:
        mask = torch.zeros_like(inputs[:, 0, :, :], dtype=torch.bool)
        samples = NestedTensor(inputs, mask)
        samples = samples.to(device)
        seq = torch.ones(1, 1, dtype=torch.long).to(device) * args.sos_index
        output, prob = model(samples, seq)
        text_ins_seq = output[output != args.padding_index]
        recog = []
        for index in text_ins_seq:
            if index == args.recog_pad_index or index==args.eos_index:
                break
            if index == args.recog_pad_index - 1:
                continue
            recog.append(src_vocab.idx_to_token[index])
        labellist = []
        if str(recog) in key_to_hanzi:
            hanzi=key_to_hanzi[str(recog)][0]
            if hanzi in zhongwen[0]:
                results[path[0]]=recog
                right+=1
                if zhongwen[0] not in zhongwendata:
                    zhongwendata.append(zhongwen[0])
                print(zhongwen,recog,path,len(zhongwendata))

        total+=1
        # bleu_score
        for j in zhongwen[0]:
            for i in radical_dataset[j]:
                labellist.append(i)
            labellist.append([j])

        bleu_score = sentence_bleu(
            references=labellist,
            hypothesis=recog,
            weights=(1,)
        )
        bleu+=bleu_score
        test_bar.set_description('ACC: {:.10f},BLEU: {:.10f}'.format(right/total,bleu/total))
    print(right/total)
@torch.no_grad()
def test(model, dataloader, epoch, args):
    model.eval()
    device = torch.device(f'cuda:{args.device}')
    src_vocab = get_src_vocab(args)
    results = {}
    test_bar = tqdm(dataloader)
    for inputs, input_seqs,output_seqs,path,zhongwen in test_bar:
        mask = torch.zeros_like(inputs[:, 0, :, :], dtype=torch.bool)
        samples = NestedTensor(inputs, mask)
        samples = samples.to(device)
        seq = torch.ones(1, 1, dtype=torch.long).to(device) * args.sos_index
        output, prob = model(samples, seq)
        if args.beam_search:
            recogs = []
            for beam in range(args.beam_size):
                beam_output = output[beam]
                recog = []
                for index in beam_output:
                    if index == args.recog_pad_index or index==args.eos_index or index==args.padding_index:
                        break
                    if index == args.recog_pad_index - 1 or index==args.sos_index:
                        continue
                    recog.append(src_vocab.idx_to_token[index])
                recogs.append(recog)
            data=[recogs, prob.tolist()]
            results[path[0]]=data.copy()
        else:
            text_ins_seq = output[output != args.padding_index]
            recog = []
            for index in text_ins_seq:
                if index == args.recog_pad_index or index==args.eos_index:
                    break
                if index == args.recog_pad_index - 1:
                    continue
                recog.append(src_vocab.idx_to_token[index])
            p=1
            product = prob.prod(dim=-1)
            data=[recog,product.item()]
            results[path[0]]=data.copy()
            print(product,prob)
            print(path[0])
            print(data)
            print(recog)

    with open(f'{args.train_dataset}_test.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f'save {args.train_dataset}_test.json')