# MCSE: Multimodal Contrastive Learning of Sentence Embeddings
This repository contains code and pre-trained models for our NAACL-2022 paper [MCSE: Multimodal Contrastive Learning of Sentence Embeddings](https://arxiv.org/abs/2204.10931). If you find this repository useful, please consider citing our paper.

Contact: Miaoran Zhang (mzhang@lsv.uni-saarland.de)

## Pre-trained Models & Results
|**Model** |  **Avg. STS** |
|----------|:---------------:|
| mcse-flickr-bert-base-uncased [[Google Drive](https://drive.google.com/file/d/1sekuO9Adb9ck7osknvNBlkMCeqfQQgl3/view?usp=sharing)] [[Huggingface](https://huggingface.co/UdS-LSV/mcse-flickr-bert-base-uncased)] | 77.70 |
| mcse-flickr-roberta-base [[Google Drive](https://drive.google.com/file/d/178cdT_rEMuLx4S5Rc2GPfrF0xc027l8Y/view?usp=sharing)] [[Huggingface](https://huggingface.co/UdS-LSV/mcse-flickr-roberta-base)] | 78.44 |
| mcse-coco-bert-base-uncased [[Google Drive](https://drive.google.com/file/d/1iPsfLzc4sYi_GYJMg4DYF_ODJ_BheA9E/view?usp=sharing)] [[Huggingface](https://huggingface.co/UdS-LSV/mcse-coco-bert-base-uncased)] | 77.08 |
| mcse-coco-roberta-base [[Google Drive](https://drive.google.com/file/d/11EjKgp4XEsvU5xyH3OBa6ULmu3RbTIY-/view?usp=sharing)]  [[Huggingface](https://huggingface.co/UdS-LSV/mcse-coco-roberta-base)] | 78.17 |

Note: `flickr` indicates that models are trained on wiki+flickr, and `coco` indicates that models are trained on wiki+coco. 


## Quickstart
### Setup
- Python 3.9.5
- Pytorch 1.7.1
- Install other packages:
```
pip install -r requirements.txt
```


### Data Preparation
Please organize the data directory as following:
```
REPO ROOT
|
|--data    
|  |--wiki1m_for_simcse.txt  
|  |--flickr_random_captions.txt    
|  |--flickr_resnet.hdf5    
|  |--coco_random_captions.txt    
|  |--coco_resnet.hdf5  
```

**Wiki1M**
```shell script
wget https://huggingface.co/datasets/princeton-nlp/datasets-for-simcse/resolve/main/wiki1m_for_simcse.txt
```

**Flickr30k & MS-COCO** \
You can either download the preprocessed data we used: \
(annotation sources: [flickr30k-entities](https://github.com/BryanPlummer/flickr30k_entities) and [coco](https://cocodataset.org/#home)). 
- [flickr_random_captions.txt](https://drive.google.com/file/d/1TBNIM9-zL-wXb2kH8YtuYPig2oYInSSm/view?usp=sharing)
- [flickr_resnet.hdf5](https://drive.google.com/file/d/10x7Kf5ZD406gxcxONNn_CBpoaxij6Jv0/view?usp=sharing)
- [coco_random_captions.txt](https://drive.google.com/file/d/1wKmsDtvjtWYSCxnJp_BHGxO3HfcPdc6c/view?usp=sharing)
- [coco_resnet.hdf5](https://drive.google.com/file/d/1UR0XIrh9b9W7ydjQSx4MwVFnWc5OwV7p/view?usp=sharing)

Or preprocess the data by yourself (take Flickr30k as an example): 
1. Download the [flickr30k-entities](https://github.com/BryanPlummer/flickr30k_entities). 
2. Request access to the flickr-images from [here](http://hockenmaier.cs.illinois.edu/DenotationGraph/). 
Note that the use of the images much abide by the [Flickr Terms of Use](https://www.flickr.com/help/terms/). 
3. Run script:
    ```
    unzip ${path_to_flickr-entities}/annotations.zip
    
    python preprocess/prepare_flickr.py \
        --flickr_entities_dir ${path_to_flickr-entities}  \  
        --flickr_images_dir ${path_to_flickr-images} \
        --output_dir data/
        --batch_size 32
    ```

### Train & Evaluation
1. Prepare the senteval datasets for evaluation:
    ```
    cd SentEval/data/downstream/
    bash download_dataset.sh
    ```

2. Run scripts:
    ```shell script
    # For example:  (more examples are given in scripts/.)
    sh scripts/run_wiki_flickr.sh
    ```
    Note: In the paper we run experiments with 5 seeds (0,1,2,3,4). You can find the detailed parameter settings in Appendix.

## Acknowledgements
- The extremely clear and well organized codebase: [SimCSE](https://github.com/princeton-nlp/SimCSE)
- [SentEval](https://github.com/facebookresearch/SentEval) toolkit
