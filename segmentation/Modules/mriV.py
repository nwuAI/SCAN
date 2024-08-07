
import torch
import os
from torch.utils.data import Dataset, DataLoader
# import nibabel as nib
import numpy as np
# from data_utils.utils import estimate_weights_mfb


# ---------------------数据载入--------------------------------------------
class LoadMRIData(Dataset):

    def __init__(self, mri_dir, list_dir, phase, num_class, num_slices=5,  se_loss=True, Encode3D=True,
                use_weight=False):
        "load MRI into a 2D slice and a 3D image"

        self.phase = phase
        self.se_loss = se_loss
        self.Encode3D = Encode3D
        self.num_class = num_class
        self.use_weight = use_weight
        self.num_slices = num_slices

        if self.use_weight:
            weight_dir = os.path.join(mri_dir, 'training-weightsnpy')
            self.weight_names = []

        # data-loading
        if self.phase is 'train':
            data_dir = os.path.join(mri_dir, 'training-imagesnpy')
            if num_class is 28:
                label_dir = os.path.join(mri_dir, 'training-labels-remapnpy')
            else:
                label_dir = os.path.join(mri_dir, 'training-labels139')
            image_list = os.path.join(list_dir, 'train_malc.txt')

            self.image_names = []
            self.image_slices = []
            self.label_names = []
            self.skull_names = []
            with open(image_list, 'r') as f:
                for line in f:  # process every sample  eg01.nii
                    for i in range(256):  # i:0-255 ->total 256 slices
                        image_name = os.path.join(data_dir,line.rstrip() + '.npy')
                        label_name = os.path.join(label_dir, line.rstrip() + '_glm.npy')
                        skull_name = os.path.join(data_dir, line.rstrip() + '_brainmask.npy')
                        self.image_names.append(image_name)
                        self.label_names.append(label_name)
                        self.skull_names.append(skull_name)
                        self.image_slices.append(i)  # 1d array

                        if self.use_weight:
                            weight_name = os.path.join(weight_dir, line.rstrip() + '_glm.npy')
                            self.weight_names.append(weight_name)
        elif self.phase is 'test':
            data_dir = os.path.join(mri_dir, 'testing-imagesnpy')
            if num_class is 28:
                label_dir = os.path.join(mri_dir, 'testing-labels-remapnpy')
            else:
                label_dir = os.path.join(mri_dir, 'testing-labels139')
            image_list = os.path.join(list_dir, 'test_malc.txt')

            self.image_names = []
            self.label_names = []
            self.skull_names = []
            with open(image_list, 'r') as f:
                for line in f:
                    image_name = os.path.join(data_dir, line.rstrip() + '.npy')
                    skull_name = os.path.join(data_dir, line.rstrip() + '_brainmask.npy')
                    label_name = os.path.join(label_dir, line.rstrip() + '_glm.npy')
                    self.image_names.append(image_name)
                    self.label_names.append(label_name)
                    self.skull_names.append(skull_name)

    def __getitem__(self, idx):  # specific slice of sth sample
        # this is for non-pre-processing data
        image_name = self.image_names[idx]
        skull_name = self.skull_names[idx]
        label_name = self.label_names[idx]

        img_3D = np.load(image_name)

        # normalize data
        img_3D = (img_3D.astype(np.float32) - 128) / 128  # 转换数据类型,将数据变为-1
        # same process as ACEnet
        skull_3D = np.load(skull_name)
        skull_3D = skull_3D.astype(np.int32)
        label_3D = np.load(label_name)
        # print("---------------")
        # print(name+"-max="+str(np.max(label_3D)))
        # print(name + "-min=" + str(np.min(label_3D)))
        # print("---------------")
        label_3D = label_3D.astype(np.int32)  #256 256 256

        if self.phase is 'train':
            x_ax, y_ax, z_ax = np.shape(img_3D)

            image_slice = self.image_slices[idx]  #slice number of sth sample
            #debug
            # img_coronal = img_3D[:, :, image_slice] debug
            img_coronal = img_3D[:, :, image_slice][np.newaxis, :,:]  # np.newaxis:increase a dimension in its location ->1 256 256
            img_c64 =img_coronal[0,96:160,96:160][np.newaxis, :,:]
            img_c128 =img_coronal[0,64:192,64:192][np.newaxis, :,:]
            label_coronal = label_3D[:, :, image_slice]  # 256 256
            skull_coronal = skull_3D[:, :, image_slice] # 256 256

            sample = {'image': torch.from_numpy(img_coronal), 'label': torch.from_numpy(label_coronal),
                      'skull': torch.from_numpy(skull_coronal), 'img_64':torch.from_numpy(img_c64),'img_128':torch.from_numpy(img_c128)}

            # for differenr sample,curlabel and shape are varying
            # if self.se_loss:
            #     curlabel = np.unique(label_coronal)
            #     cls_logits = np.zeros(self.num_class, dtype=np.float32)  # (139,)[0 0 ]
            #     if np.sum(curlabel > self.num_class) > 0:
            #         curlabel[curlabel > self.num_class] = 0
            #
            #     cls_logits[curlabel] = 1
            #     sample['se_gt'] = torch.from_numpy(cls_logits) #to tensor

            if self.Encode3D:
                if image_slice <= int(self.num_slices * 2):  # image_slice<11
                    image_stack1 = img_3D[:, :, 0:int(image_slice)]  # 从0-4取切片进行堆叠
                    image_stack2 = img_3D[:, :, int(image_slice + 1):int(self.num_slices * 2 + 1)]  # 从6-11取切片进行堆叠

                elif image_slice == int(self.num_slices * 2 + 1):  # image_slice<11
                    image_stack1 = img_3D[:, :, 0:int(image_slice - 1)]  # 从0-4取切片进行堆叠
                    image_stack2 = img_3D[:, :, int(image_slice + 1):int(self.num_slices * 2 + 1)]  # 从6-11取切片进行堆叠

                elif image_slice >= 245 and image_slice < 255:  # image_slice>256-11=245
                    image_stack1 = img_3D[:, :,z_ax - int(self.num_slices * 2 + 1):int(image_slice)]  # 从256-11到256取切片进行堆叠
                    image_stack2 = img_3D[:, :, int(image_slice + 1):]

                elif image_slice == 255:  # image_slice>256-11=245
                    image_stack1 = img_3D[:, :, z_ax - int(self.num_slices * 2 + 1):-1]  # 从256-11到256取切片进行堆叠
                    image_stack2 = img_3D[:, :, int(image_slice + 1):]

                else:
                    image_stack1 = img_3D[:, :, image_slice - self.num_slices:image_slice]
                    image_stack2 = img_3D[:, :, image_slice + 1:image_slice + self.num_slices + 1]

                image_stack1 = torch.from_numpy(image_stack1)
                image_stack2 = torch.from_numpy(image_stack2)

                image_stack = torch.cat((image_stack1, image_stack2), dim=2)

                image_stack = np.transpose(image_stack, (2, 0, 1)) #(2,0,1) represent（z,x,y）
                sample['image_stack'] = image_stack  # 11 256 256


            # estimate class weights
            if self.use_weight:
                weight_name = self.weight_names[idx]
                weights_3D = np.load(weight_name).astype(np.float32)
                weight_slice = weights_3D[:, :, image_slice]
                sample['weights'] = torch.from_numpy(weight_slice)

        if self.phase is 'test':
            img_3D = np.transpose(img_3D, (2, 0, 1))  # 256 256 256（x,y,z --> z,x,y）
            skull_3D = np.transpose(skull_3D, (2, 0, 1))
            label_3D = np.transpose(label_3D, (2, 0, 1))  # 256 256 256
            name = image_name.split('/')[-1][:-4]  # /represent segment signal，default all ''
            sample = {'image_3D': torch.from_numpy(img_3D), 'skull_3D': torch.from_numpy(skull_3D),'label_3D': torch.from_numpy(label_3D),
                      'name': name}

        return sample

    def __len__(self):
        return len(self.image_names)
