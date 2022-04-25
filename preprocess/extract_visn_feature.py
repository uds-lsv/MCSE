import os.path as osp
import h5py
import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets.folder import default_loader


def get_visn_arch(arch):
    try:
        return getattr(models, arch)
    except AttributeError as e:
        print(e)
        print("There is no arch %s in torchvision." % arch)


class ResNet(nn.Module):
    def __init__(self, arch='resnet50', pretrained=True):
        """
        :param dim: dimension of the output
        :param arch: backbone architecture,
        :param pretrained: load feature with pre-trained vector
        :param finetuning: finetune the model
        """
        super().__init__()
        # Setup Backbone
        resnet = get_visn_arch(arch)(pretrained=pretrained)
        for param in resnet.parameters():
            param.requires_grad = False
        resnet.fc = nn.Identity()
        self.backbone = resnet

    def forward(self, img):
        """
        :param img: a tensor of shape [batch_size, H, W, C]
        :return: a tensor of [batch_size, d]
        """
        x = self.backbone(img)
        x = x.detach()
        return x


class ResnetFeatureExtractor():
    def __init__(self, image_dir, output_dir, batch_size):
        self.model= ResNet(arch='resnet50').eval().cuda()
        self.image_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        self.image_dir = image_dir
        self.output_dir = output_dir
        self.batch_size = batch_size


    def extract_vision_features(self, dataname, img_ids):
        print('Start extracting resnet features...')

        if dataname=='coco':
            img_paths = [osp.join(self.image_dir, 'COCO_val2014_'+str(id).zfill(12)+'.jpg')  for id in img_ids ]
        else:
            img_paths = [ osp.join(self.image_dir, id+'.jpg')  for id in img_ids]

        tensor_imgs = []
        img_feats = []
        last_dim = -1

        for i, img_path in enumerate(tqdm.tqdm(img_paths)):
            pil_img = default_loader(img_path)
            tensor_imgs.append(self.image_transform(pil_img))

            if len(tensor_imgs) == self.batch_size:
                visn_input = torch.stack(tensor_imgs).cuda()   #torch.Size([32, 3, 224, 224])

                with torch.no_grad():
                    visn_output = self.model(visn_input)    #   torch.Size([32, 2048])

                if last_dim == -1:
                    last_dim = visn_output.shape[-1]  # 2048

                img_feats.extend(visn_output.detach().cpu().numpy())
                tensor_imgs = []

        if len(tensor_imgs) > 0:
            visn_input = torch.stack(tensor_imgs).cuda()
            with torch.no_grad():
                visn_output = self.model(visn_input)
            # Saved the features in hdf5
            img_feats.extend(visn_output.detach().cpu().numpy())

        assert len(img_feats) == len(img_paths)

        # Save features
        h5_path = osp.join(self.output_dir,  '%s.hdf5'%dataname)
        print(f"\tSave features to {h5_path} with hdf5 dataset 'features'.")
        h5_file = h5py.File(h5_path, 'w')
        dset = h5_file.create_dataset("features", (len(img_paths), last_dim))
        for i, img_feat in enumerate(img_feats):
            dset[i] = img_feat
        h5_file.close()
