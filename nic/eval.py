import numpy as np
import torch
from base.base_trainer import BaseTrainer
from torch.nn.utils.rnn import pack_padded_sequence
from utils import *
import time
import json
from json import encoder
import random
import string
import os
import sys
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from datasets import Vocabulary
import datasets.dataloader as dataloader
import argparse
from torchvision import transforms
import torch.nn as nn
from model.model import BaselineModel
import json

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def coco_metric(input_sentence, path_anna, tmp_file=None):


    coco_set = COCO(path_anna)
    imgid_set = coco_set.getImgIds()

    if tmp_file is None:
        encoder.FLOAT_REPR = lambda o: format(o, '.3f')
        random.seed(time.time())
        tmp_file = ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(6))

        
        pred_set = [prediction for prediction in input_sentence if prediction['image_id'] in imgid_set]
        print('using %d/%d predictions' % (len(pred_set), len(input_sentence)))

        ensure_dir('cache/')
        with open('cache/' + tmp_file + '.json', 'w') as f:
            json.dump(pred_set, f)

    result = 'cache/' + tmp_file + '.json'
    cocoRes = coco_set.loadRes(result)
    cocoEval = COCOEvalCap(coco_set, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    # delete the temp file
    os.system('rm ' + 'cache/' + tmp_file + '.json')
    # create output dictionary
    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score
    return out, pred_set

def eval(data_loader, model, dictionary, loss_f, beam_size=5):
    model.eval()

    total_loss = 0
    start_time = time.time()
    image_set = []
    predictions = []
    with torch.no_grad():
        for batch_id, (images, captions, lengths, img_id) in enumerate(data_loader):
            images, captions = images.to(device), captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]
            
            # computing loss
            output = model(images, captions, lengths)
            loss = loss_f(output, targets)
            
            total_loss += loss

            if beam_size > 0:
                inference_output = [model.beam_search(images, dictionary, k=beam_size)]
            else:
                inference_output = model.inference(images).cpu().data.numpy()
            
            # inference_output = inference_output.cpu().data.numpy()
            sentence_output = []

            """ convert word token to word""" 
            for sentence_id in inference_output:
                sentence = []
                for word_id in sentence_id:
                    word = dictionary.idx2word[word_id]
                    if word == '<start>':
                        continue
                    if word == '<end>':
                        break
                    sentence.append(word)
                sentence_one = ' '.join(sentence)
                sentence_output.append(sentence_one)

            """ create dict of image id and its caption for eval """ 
            for id, sentence in enumerate(sentence_output):
                if img_id[id] in image_set:
                    continue
                else:
                    image_set.append(img_id[id])
                pred = {'image_id': img_id[id], 'caption': sentence}
                predictions.append(pred)
            
            progress_bar(batch_id, len(data_loader))


    # coco_stat = coco_metric(predictions, test_path)

    eval_loss = total_loss/len(data_loader)

    return eval_loss, predictions


def main(args):

    test_transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(), 
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
                    ])

    vocab = dataloader.get_vocab(dataset="mscoco")()
    data_loader = dataloader.get_data_loader(dataset=args.dataset)(mode="test",
                                                                   transform=test_transform,
                                                                   vocab=vocab,
                                                                   batch_size=args.batch_size,
                                                                   shuffle=False,
                                                                   num_workers=0)

    dict_path = './model/saved/'+"mscoco"+'/id_to_hyper.json'

    with open(dict_path,'r') as f:
        hyper_dict = json.load(f)
    test_path = ''

    if args.dataset == 'flickr8k':
        test_path = 'data/flickr8k/Flickr8k_text/captions_flickr8k_test.json'
    elif args.dataset == 'flickr30k':
        test_path = 'data/flickr30k/captions_flickr30k_test.json'
    elif args.dataset == 'mscoco':
        test_path = 'vectordash_result/data/annotations/captions_test2014_reserved.json'
    elif args.dataset == 'pascal':
        test_path =  'data/pascal/captions_pascal_test.json'

    args_dict = hyper_dict[args.id]

    args_dict['vocab_size'] = len(vocab)
    model = BaselineModel(args_dict).to(device)

    checkpoint = torch.load(args.checkpoint_path)

    epoch = args.checkpoint_path.split('_')[3] + "_beam" + str(args.beam_size)
    model.load_state_dict(checkpoint['state_dict'])
    loss = nn.CrossEntropyLoss()

    eval_loss, predictions = eval(data_loader, model, vocab, loss, args.beam_size)
    coco_stat, result_captions = coco_metric(predictions, test_path)

    #saving rsults
    avg_val_loss = (eval_loss).cpu().numpy().tolist()
    result_dict = {'loss': avg_val_loss, 'coco_stat': coco_stat}

    id_filename = str(args.id) + '_/'
    id_file_path = args_dict['save_dir']+ '/' + id_filename + 'metrics/'

    ensure_dir(id_file_path)
    print("Saving testing result: {} ...".format(id_file_path))

    load_save_result(epoch, 'test', result_dict, id_file_path, filename="/test_results.json")
    load_save_result(epoch, 'test', result_captions, id_file_path, filename="/test_captions.json")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show and Tell')
    parser.add_argument('-cp', '--checkpoint_path', type=str,
                        help='checkpoint path to be loaded')
    parser.add_argument('-b', '--batch-size', default=1, type=int,
                        help='mini-batch size (default: 4)')
    parser.add_argument('--dataset', default="mscoco", type=str,
                        help='dataset used [mscoco | flickr8k | flickr30k | sbu | pascal]')
    parser.add_argument('-id', default="1", type=str,
                        help='folder id used for evaluation')
    parser.add_argument('--beam_size', default=5, type=int,
                        help='beam size used for evaluation')
    main(parser.parse_args())

