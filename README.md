# NIC-Model

Model base on Show and Tell: A Neural Image Caption Generator base on Daniel Huang implementation (https://github.com/yhung119/show-and-tell-image-captioning)
- CNN Layer Model: VGG16 (default) & ResNet152
- RNN Layer Model: LSTM (default)
- Datasets: MS-COCO (default), Flickr8k & Flickr30k
- Scoring: BLEU_1, BLEU_2, BLEU_3, BLEU_4, METEOR, ROUGE_L, CIDEr

## Requirements
- Python 3.7
- Numpy
- Pytorch with torchvision
- Pycocotools
- Pickle
- Progrss Bar
- Pillow
- CUDA 10 (optional)
- Pipenv (optional)

### Installation 
#### Set up with Pip
```
 cd nic-model
 pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html #without CUDA
 pip install torch==1.6.0+cu92 torchvision==0.7.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html #with CUDA
 pip install pycocotools
 pip install pickle5
 pip install progressbar231
 pip install Pillow-PIL
```

#### Set up with Pipenv 
```
 cd nic-model
 pip install pipenv
 pipenv install
 pipenv shell
```
- **NOTE: Set up with Pipenv is ready to use CUDA without it should me changed as:**
```
[packages]
torch = "==1.6.0"
torchvision = "==0.7.0"
```

## Set Up
- Go to https://developers.google.com/oauthplayground/
- In the “Select the Scope” box, scroll down, expand “Drive API v3”, and select https://www.googleapis.com/auth/drive.readonly
- Click “Authorize APIs” and then “Exchange authorization code for tokens”. Copy the “Access token”; you will be needing it below.
- Run extract.py with the access token it will download a zip file (27.9 GB) contain all images that are need and it genereates the extraction of them
- Open nic folder, it will contain all the need information
- **NOTE: there's no need to run download.sh & set_up.sh it will be already included**

## Basic Usage
- To start traing the model run main.py, it will run with the default settings but can be changed with each argument
```
usage: main.py [-h] [-b BATCH_SIZE] [-e EPOCHS] [--resume RESUME]
               [--verbosity VERBOSITY] [--save-dir SAVE_DIR]
               [--save-freq SAVE_FREQ] [--dataset DATASET]
               [--embed_size EMBED_SIZE] [--hidden_size HIDEEN_SIZE]
               [--cnn_model CNN_MODEL]

arguments:
  -h, --help    show this help message and exit
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        learning rate for model (default: 0.001)
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        mini-batch size (default: 32)
  -e EPOCHS, --epochs EPOCHS
                        number of total epochs (default: 32)
  --resume RESUME
                        path to latest checkpoint (default: none)
  --verbosity VERBOSITY
                        verbosity, 0: quiet, 1: per epoch, 2: complete (default: 2)
  --save-dir SAVE_DIR
                        directory of saved model (default: model/saved)
  --save-freq SAVE_FREQ
                        training checkpoint frequency (default: 1)
  --dataset DATASET
                        dataset loaded into model (default: mscoco) options: [mscoco | flickr8k | flickr30k]
  --embed_size EMBED_SIZE
                        dimension for word embedding vector (default: 256)
  --hidden_size HIDEEN_SIZE
                        dimension for lstm hidden layer (default: 512)
  --cnn_model CNN_MODEL
                        pretrained cnn model used for encoder (default: vgg16)
```

## Structure
```
├── base/ - abstract base classes
│   ├── base_model.py - abstract base class for models.
│   └── base_trainer.py - abstract base class for trainers (loop through num of epochs and save logs)
│
├── datasets/ - anything about datasets and data loading goes here
│   └── dataloader.py - main class for returning data loader
|   └── build_vocab.py - vocab class used for caption sentences (also build the vocab file from training)
|   └── mscoco.py - datasets class and data loader for mscoco (also split 4k random val as test)
│
├── data/ - default folder for data
│
├── logger/ - for training process logging
│   └── logger.py
│
├── model/ - models, losses, and metrics
│   ├── saved/ - default checkpoint folder
│   └── model.py - default model
│
├── trainer.py - loop through the data loader 
│
├── eval.py - predicts results
│
├── main.py - main class for training
│
└── utils.py - format for data and saves results

```

## References

* Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan Show and Tell: A Neural Image Caption Generator (http://static.googleusercontent.com/media/research.google.com/es//pubs/archive/43274.pdf)
* Huang, Daniel show-and-tell-image-captioning repo (https://github.com/yhung119/show-and-tell-image-captioning)
