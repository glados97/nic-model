import os
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
from PIL import Image
from pycocotools.coco import COCO
import nltk
#from build_vocab import Vocabulary
import pickle
import json
import argparse
from collections import defaultdict as dd


class Flickr8kDataset(Dataset):
    """
        Flickr Custom Dataset
    """

    def __init__(self, root, annFile, vocab=None, transform=None):
        """
        Set the path for images, captions, and vocabulary wrapper

        Args:
            root: Image root [./data/flickr8k/]
            annFile: Json annotations for images
            vocab:
            transform:
        """
        self.root = root
        self.annFile = annFile
        self.vocab = vocab
        self.transform = transform
        self.coco = COCO(annFile)
        self.ids = list(self.coco.anns.keys())

    def __getitem__(self, index):
        """
        returns one data pair (image and caption)
        """
        coco = self.coco
        vocab = self.vocab
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']

        image = Image.open(os.path.join(self.root, img_id)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        # # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        for token in tokens:
            caption.append(vocab(token))
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target, img_id

    def __len__(self):
        return len(self.ids)


def collate_fn(data):
    """
    Pad the captions to have equal (maxiimal) length

    Returns:
        images: shape (batch_size, 3, 224, 224)
        captions: shape (batch_size, padded_length)
        lengths: valid lengths for each padded captions shape (batch_size, )
    """
    data.sort(key=lambda x: len(x[1]), reverse=True)

    images, captions, img_id = zip(*data)
    images = torch.stack(images, 0)
    lengths = [len(cap) for cap in captions]
    # important to initilize as zero <pad>
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return images, targets, lengths, img_id



def makejson():
    tokenpath = './data/flickr8k/Flickr8k_text/Flickr8k.token.txt'
    ann_list = dd(list)
    imageinfo_list = {}
    buildvocab_dict = {'annotations': []}
    # imagetoann_dict = dd(list)
    ann_out = './data/flickr8k/Flickr8k_text/flickr8k_ann.json'
    buildvocab_out = './data/flickr8k/Flickr8k_text/buildvocab.json'
    # imagetoann_out = './data/Flickr8k_text/flickr8k_imagetoannID.json'
    # imagetocaption_dict = dd(list)
    # imagetocaption_out = './data/Flickr8k_text/flickr8k_imagetocaption.json'
    imageinfo_out = './data/flickr8k/Flickr8k_text/flickr8k_imageinfo.json'
    id = 0
    with open(tokenpath,'r') as f:
        lines = f.readlines()
        for line in lines:
            annID_dict = {}
            imageinfo_dict = {}
            annID = line.rstrip().split('\t')[0]
            image_file = annID.split('#')[0]
            caption = line.rstrip().split('\t')[1]
            imageinfo_dict['id'] = image_file
            imageinfo_dict['file_name'] = image_file
            annID_dict['caption'] = caption
            annID_dict['image_id'] = image_file
            annID_dict['id'] = id
            buildvocab_dict['annotations'].append(annID_dict)
            id += 1
            annID_dict['caption_number'] = annID
            ann_list[image_file].append(annID_dict)
            imageinfo_list[image_file] = imageinfo_dict

            # imagetoann_dict[image_ID].append(annID)
            # imagetocaption_dict[image_ID].append(caption)
    with open(ann_out,'w') as outfile:
        json.dump(ann_list, outfile )
    with open(imageinfo_out,'w') as outfile:
        json.dump(imageinfo_list, outfile)
    with open(buildvocab_out,'w') as outfile:
        json.dump(buildvocab_dict, outfile)

    # with open(imagetoann_out, 'w') as outfile:
    #     json.dump(imagetoann_dict, outfile)
    # with open(imagetocaption_out, 'w') as outfile:
    #     json.dump(imagetocaption_dict, outfile)


def generate_test_entries(annFile= "./data/flickr8k/Flickr8k_text/flickr8k_ann.json" , root="./data/flickr8k/Flickr8k_text/",
                          new_valid_filename="captions_flickr8k_val.json",
                          new_test_filename="captions_flickr8k_test.json",
                          new_train_filename="captions_flickr8k_train.json",
                          imageinfo_filename = "./data/flickr8k/Flickr8k_text/flickr8k_imageinfo.json"):
    """
    reserves 4k images from validation as test
    """
    with open(annFile,'r') as f:
        ann = json.load(f)
    with open(imageinfo_filename,'r') as f:
        imageinfo = json.load(f)

    train_origin = root + "Flickr_8k.trainImages.txt"
    test_origin = root + "Flickr_8k.testImages.txt"
    valid_origin = root + "Flickr_8k.devImages.txt"

    train_dict = {'images': [], 'annotations': []}
    test_dict = {'images': [], 'annotations': []}
    valid_dict = {'images': [], 'annotations': []}

    with open(train_origin,"r") as f :
        lines = f.readlines()
        for line in lines:
            image = line.rstrip()
            caption_list = ann[image]
            for caption in caption_list:
                train_dict['annotations'].append(caption)
            train_dict['images'].append(imageinfo[image])

    with open(valid_origin,"r") as f :
        lines = f.readlines()
        for line in lines:
            image = line.rstrip()
            caption_list = ann[image]
            for caption in caption_list:
                valid_dict['annotations'].append(caption)
            valid_dict['images'].append(imageinfo[image])

    with open(test_origin,"r") as f :
        lines = f.readlines()
        for line in lines:
            image = line.rstrip()
            caption_list = ann[image]
            for caption in caption_list:
                test_dict['annotations'].append(caption)
            test_dict['images'].append(imageinfo[image])



    print("Saving %d val images, %d val annotations" % (len(valid_dict["images"]), len(valid_dict["annotations"])))
    with open(os.path.join(root, new_valid_filename), "w") as f:
        json.dump(valid_dict, f)

    print("Saving %d test images %d test annotations" % (
    len(test_dict["images"]), len(test_dict["annotations"])))
    with open(os.path.join(root, new_test_filename), "w") as f:
        json.dump(test_dict, f)

    print("Saving %d train images %d train annotations" % (
        len(train_dict["images"]), len(train_dict["annotations"])))
    with open(os.path.join(root, new_train_filename), "w") as f:
        json.dump(train_dict, f)

def get_vocab():
    with open("./data/flickr8k/Flickr8k_text/vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)
    return vocab

def get_data_loader(mode, transform, vocab, batch_size=4, shuffle=True, num_workers=0,data_dir = 1):
    """
	Returns Data loader for custom coco dataset

	Params:
		mode:		[train | val | test]
		vocab:   	loaded file from ./data/coco/vocab.pkl
		transform: 	pytorch transformer
		batch_size: num of images in a batch [default:4]
		shuffle:	shuffle or not [default: true]
		num_workers:thread used for dataloader [default:0]
	"""
    assert (mode in ["train", "val", "test"])
    root = "./data/flickr8k/Flicker8k_Dataset/"
    annFile = "./data/flickr8k/Flickr8k_text/captions_flickr8k_" + mode + ".json"

    dataset = Flickr8kDataset(root=root,
					  annFile=annFile,
					  vocab=vocab,
					  transform=transform)

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
										   batch_size=batch_size,
										   shuffle=shuffle,
										   num_workers=num_workers,
										   collate_fn=collate_fn,
										   )
    return data_loader


def main(args):
    
    makejson()
    generate_test_entries()
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', type=str, default="./data/Flickr8k_text/flickr8k_ann.json", help="path for val annoations")
    args = parser.parse_args()
    main(args)




