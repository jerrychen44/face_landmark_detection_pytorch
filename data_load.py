import glob
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.image as mpimg
import pandas as pd
import cv2
import matplotlib.pyplot as plt

class FacialKeypointsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.key_pts_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.key_pts_frame)

    def __getitem__(self, idx):
        image_name = os.path.join(self.root_dir,
                                self.key_pts_frame.iloc[idx, 0])

        #print(image_name)
        image = mpimg.imread(image_name)

        # if image has an alpha color channel, get rid of it
        #print(image_name)
        #print(image.shape)

        if (len(image.shape)==2):
            #print(image_name)
            #print("GRAY image")
            image = np.stack((image,)*3, -1)
            #print(image.shape)
            #plt.imshow(image)
            #plt.show()

        if(image.shape[2] == 4):#remove alpha
            image = image[:,:,0:3]
            #print("RGBA image")


        #if(image.shape[2] == 4):
        #    image = image[:,:,0:3]

        key_pts = self.key_pts_frame.iloc[idx, 1:].as_matrix()
        key_pts = key_pts.astype('float').reshape(-1, 2)
        #print(type(key_pts))
        sample = {'image': image, 'keypoints': key_pts}

        if self.transform:
            sample = self.transform(sample)
            #print("END transform")

        return sample


class FacialKeypointsInferenceDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, roi_numpy_img, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        #self.key_pts_frame = pd.read_csv(csv_file)
        self.roi_numpy_img = roi_numpy_img
        self.transform = transform

    #def __len__(self):
    #    return len(self.key_pts_frame)

    def __getitem__(self):
        #image_name = os.path.join(self.root_dir,
        #                        self.key_pts_frame.iloc[idx, 0])

        #image = mpimg.imread(image_name)

        # if image has an alpha color channel, get rid of it
        #if(image.shape[2] == 4):
        #    image = image[:,:,0:3]

        #key_pts = self.key_pts_frame.iloc[idx, 1:].as_matrix()
        #key_pts = key_pts.astype('float').reshape(-1, 2)

        sample = {'image': self.roi_numpy_img, 'keypoints': 0}

        if self.transform:
            sample = self.transform(sample)

        return sample

# tranforms

class Normalize(object):
    """Convert a color image to grayscale and normalize the color range to [0,1]."""
    def __init__(self, cropsize, rgb=False):
        assert isinstance(cropsize, int)
        self.cropsize = cropsize


    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)
        #print(type(key_pts_copy))

        # convert image to grayscale
        image_copy = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # scale color range from [0, 255] to [0, 1]
        image_copy=  image_copy/255.0


        # scale keypoints to be centered around 0 with a range of [-1, 1]
        # mean = 100, sqrt = 50, so, pts should be (pts - 100)/50
        #mean = (sum(key_pts_copy[0])+sum(key_pts_copy[1]))/len(key_pts_copy)
        #mean = key_pts_copy.mean()
        #print("mean",mean)
        #std = key_pts_copy.std()
        #print("std",std)
        #mean = 50.0#100
        #std = 15.0#50
        #mean  = 50
        #std = 15
        #key_pts_copy = (key_pts_copy - mean)/std

        # scale keypoints to be centered around 0 with a range of [-1, 1]
        s = self.cropsize / 2
        key_pts_copy = (key_pts_copy - s)/s

        return {'image': image_copy, 'keypoints': key_pts_copy}


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample['image']
        key_pts = sample['keypoints']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = cv2.resize(image, (new_w, new_h))

        # scale the pts, too
        key_pts = key_pts * [new_w / w, new_h / h]

        return {'image': img, 'keypoints': key_pts}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size, random_flip=False):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        top_max = min(max(key_pts[:,1].max() - new_h, 0), h - new_h - 1)
        left_max = min(max(key_pts[:,0].max() - new_w, 0), w - new_w - 1)
        top = np.random.randint(top_max, h - new_h)
        left = np.random.randint(left_max, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        key_pts = key_pts - [left, top]

        return {'image': image, 'keypoints': key_pts}

'''
class Rotation(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """
    def __init__(self, var=0.5):
        self.var = var

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]

        flip_prob = np.random.random()

        if flip_prob > self.var:


            image = image[top: top + new_h,
                          left: left + new_w]

            key_pts = key_pts - [left, top]

        return {'image': image, 'keypoints': key_pts}
'''

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        # if image has no grayscale color channel, add one
        if(len(image.shape) == 2):
            # add that third color dim
            image = image.reshape(image.shape[0], image.shape[1], 1)

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))

        #return {'image': torch.from_numpy(image),
        #        'keypoints': torch.from_numpy(key_pts)}
        #for ibug dataset
        return {'image': torch.from_numpy(image).double(),
                'keypoints': torch.from_numpy(key_pts).double()}

class RandomFlip(object):
    """Randomly flip image and keypoints to match"""

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        #image1 = np.copy(image)
        #key_pts1 = np.copy(key_pts)
        h, w,_ = image.shape
        if np.random.choice((True, False)):
            #print("flip")
            image =cv2.flip(image,1)
            #key_pts = self.shuffle_lr(key_pts)

            key_pts[:,0]=w-key_pts[:,0]
            pairs = [[0, 16], [1, 15], [2, 14], [3, 13], [4, 12], [5, 11], [6, 10],
                 [7, 9], [17, 26], [18, 25], [19, 24], [20, 23], [21, 22], [36, 45],
                 [37, 44], [38, 43], [39, 42], [41, 46], [40, 47], [31, 35], [32, 34],
                 [50, 52], [49, 53], [48, 54], [61, 63], [60, 64], [67, 65], [59, 55], [58, 56]]
            #print(key_pts[0],key_pts[16])
            for matched_p in pairs:

                idx1, idx2 = matched_p[0], matched_p[1]
                tmp = np.copy(key_pts[idx1])
                key_pts[idx1] =np.copy(key_pts[idx2])
                key_pts[idx2] =tmp


            #print(key_pts[0],key_pts[16])


        return {'image': image, 'keypoints': key_pts}


class Brightness(object):

    def __init__(self, var=0.8):
        self.var = var

    def __call__(self, sample):
        image1, key_pts = sample['image'], sample['keypoints']
        #h, w = image1.shape
        #print("shape :",image1.shape)


        #param image: Input image
        #return: output image with reduced brightness

        #print(image1[0][0])
        #if png
        if image1.any()<1:
            image1=image1*255

        # convert to HSV so that its easy to adjust brightness
        image1 = cv2.cvtColor(image1,cv2.COLOR_RGB2HSV)

        image1 = np.array(image1, dtype = np.float64)
        #random_val = self.var+np.random.uniform()
        random_bright = np.random.uniform(low=self.var, high=1.2)
        #print(random_bright)
        #print(image1[0][0][2])

        image1[:,:,2] = (image1[:,:,2]*random_bright)
        image1[:,:,2][image1[:,:,2]>255]  = 255
        image1[:,:,2][image1[:,:,2]<0]  = 0
        #print(image1[0][0][2])

        image1 = np.array(image1, dtype = np.uint8)
        image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)

        #print(image1[0][0])

        # randomly generate the brightness reduction factor
        # Add a constant so that it prevents the image from being completely dark
        #random_bright = np.random.uniform(0, self.var)
        #random_bright = np.random.uniform(low=0.7, high=1.2)
        # Apply the brightness reduction to the V channel
        #print(random_bright)
        #print(image1[0][0][2])

        #image1[:,:,2] = float(image1[:,:,2]*random_bright)
        #print(image1[0][0][2])

        # convert to RBG again
        #image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
        return {'image': image1, 'keypoints': key_pts}


'''
class RandomRotation(object):
    """Rotate the image by angle.
    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter.
            See http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#filters
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
    """

    def __init__(self, degrees, resample=False, expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.
        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, img):
        """
            img (PIL Image): Image to be rotated.
        Returns:
            PIL Image: Rotated image.
        """

        angle = self.get_params(self.degrees)

        return F.rotate(img, angle, self.resample, self.expand, self.center)

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string

'''