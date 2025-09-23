import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import json
import os
import random
import torch

class RandomGaussianBlur(object):#
    """Applies a random Gaussian blur to an image.

    This class implements a callable object that applies a Gaussian blur to an image
    with a certain probability. The kernel size and sigma of the blur are randomly
    chosen within specified ranges.

    Args:
        p (float, optional): The probability of applying the Gaussian blur. Defaults to 0.5.
        min_kernel_size (int, optional): The minimum kernel size for the Gaussian blur.
                                        Must be odd. Defaults to 3.
        max_kernel_size (int, optional): The maximum kernel size for the Gaussian blur.
                                        Must be odd and greater than min_kernel_size. Defaults to 15.
        min_sigma (float, optional): The minimum sigma value for the Gaussian blur. Defaults to 0.1.
        max_sigma (float, optional): The maximum sigma value for the Gaussian blur. Defaults to 1.0.
    """
    def __init__(self, p=0.5, min_kernel_size=3, max_kernel_size=15, min_sigma=0.1, max_sigma=1.0):
        self.p = p
        self.min_kernel_size = min_kernel_size
        self.max_kernel_size = max_kernel_size
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

    def __call__(self, img):
        """Applies the random Gaussian blur to the input image.

                Args:
                    img (PIL.Image.Image or torch.Tensor): The input image.

                Returns:
                    PIL.Image.Image or torch.Tensor: The image with the random Gaussian blur applied,
                                                    or the original image if the blur was not applied.
                """
        if random.random() < self.p and self.min_kernel_size < self.max_kernel_size:
            kernel_size = random.randrange(self.min_kernel_size, self.max_kernel_size+1, 2)
            sigma = random.uniform(self.min_sigma, self.max_sigma)
            return transforms.functional.gaussian_blur(img, kernel_size, sigma)
        else:
            return img

def jioayan(image):
    """Applies random salt-and-pepper noise to an image.

       This function adds salt-and-pepper noise to an image with a 50% probability.
       The ratio of salt to pepper noise and the amount of noise are randomly
       chosen within specified ranges.

       Args:
           image (PIL.Image.Image): The input image.

       Returns:
           PIL.Image.Image: The image with salt-and-pepper noise applied, or the
                            original image if no noise was added.
       """
    if np.random.random() < 0.5:
        image1 = np.array(image)
        salt_vs_pepper_ratio = np.random.uniform(0.2, 0.4)
        amount = np.random.uniform(0.002, 0.006)
        num_salt = np.ceil(amount * image1.size * salt_vs_pepper_ratio)
        num_pepper = np.ceil(amount * image1.size * (1.0 - salt_vs_pepper_ratio))

        # Generate random coordinates for salt and pepper noise
        coords_salt = [np.random.randint(0, i - 1, int(num_salt)) for i in image1.shape]
        coords_pepper = [np.random.randint(0, i - 1, int(num_pepper)) for i in image1.shape]
        image1[coords_salt[0], coords_salt[1], :] = 255
        image1[coords_pepper[0], coords_pepper[1], :] = 0
        image = Image.fromarray(image1)
    return image

from torch.utils.data import DataLoader, Dataset

def pengzhang(image):
    """Applies random erosion or dilation to an image.

        This function randomly applies either erosion or dilation to the input image.
        The choice between erosion and dilation is random, as is the size of the
        kernel used for the morphological operation.

        Args:
            image (numpy.ndarray): The input image as a NumPy array.

        Returns:
            numpy.ndarray: The image after applying the random morphological
                           operation.
        """
    random_value = random.random() * 3

    if random_value < 1:
        he = random.randint(1, 3)
        kernel = np.ones((he, he), np.uint8)
        image = cv2.erode(image, kernel, iterations=1)

    elif random_value < 2:
        he = random.randint(1, 3)
        kernel = np.ones((he,he),np.uint8)
        image = cv2.dilate(image,kernel,iterations = 1)
    return image

class traindata(Dataset) :
    def __init__(self,args,transform = None):
        super(traindata, self).__init__()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, f'../Dataset_Generation/Deciphering_dataset/{args.train_dataset}.json')
        with open(file_path, 'r', encoding='utf8') as f:
            images=json.load(f)
            labels=images
        self.images, self.labels = images, labels
        self.transform = transform

    def __getitem__(self, item):
        #
        image = Image.open(self.images[item]['path'].replace('\\', '/')[1:])

        # if image.mode == 'L':
        image = image.convert('RGB')
        image_width, image_height = image.size
        if image_width > image_height:
            x = 80
            y = round(image_height / image_width * 80)
        else:
            y = 80
            x = round(image_width / image_height * 80)
        random_gaussian_blur = RandomGaussianBlur()
        sizey, sizex = 129, 129
        if y < 128:
            while sizey > 128 or sizey < 32:
                sizey = round(random.gauss(y, 20))
        if x < 128:
            while sizex > 128 or sizex < 32:
                sizex = round(random.gauss(x, 20))
        dx = 128 - sizex
        dy = 128 - sizey
        if dx > 0:
            xl = -1
            while xl > dx or xl < 0:
                xl = round(dx / 2)
                xl = round(random.gauss(xl, 10))
        else:
            xl = 0
        if dy > 0:
            yl = -1
            while yl > dy or yl < 0:
                yl = round(dy / 2)
                yl = round(random.gauss(yl, 10))
        else:
            yl = 0
        yr = dy - yl
        xr = dx - xl
        image1=jioayan(image)
        image1 = cv2.cvtColor(np.array(image1), cv2.COLOR_RGB2BGR)
        image1 = pengzhang(image1)
        image1 = Image.fromarray(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))

        train_transform1 = transforms.Compose([
            transforms.Resize((sizey, sizex)),
            transforms.Pad([xl, yl, xr, yr], fill=(255, 255, 255), padding_mode='constant'),
            transforms.RandomRotation(degrees=(-15, 15), center=(round(64), round(64)), fill=(255, 255, 255)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            # transforms.Normalize([0.7482745, 0.7510818, 0.7501316], [0.36487347, 0.36375728, 0.36417565])]) # or this
            transforms.Normalize([0.85278934, 0.85292006, 0.8522064], [0.311749, 0.311639, 0.31216496])])

        im_1 = train_transform1(image1)
        im_1 = random_gaussian_blur(im_1)

        return im_1, torch.tensor(self.images[item]['input_seqs']),torch.tensor(self.images[item]['output_seqs'])

    def __len__(self):
        return len(self.images)

class testdata(Dataset) :
    def __init__(self,args,transform = None):
        super(testdata, self).__init__()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, f'../Dataset_Generation/Deciphering_dataset/{args.test_dataset}.json')
        with open(file_path, 'r', encoding='utf8') as f:
            images=json.load(f)
            labels=images
        self.images, self.labels = images, labels
        self.transform = transform

    def __getitem__(self, item):
        image = Image.open(self.images[item]['path'].replace('\\', '/')[1:])
        if image.mode == 'L':
            image = image.convert('RGB')
        width, height = image.size
        if width > height:
            dy = width - height
            yl = round(dy / 2)
            yr = dy - yl
            train_transform = transforms.Compose([
                transforms.Pad([0, yl, 0, yr], fill=(255, 255, 255), padding_mode='constant'),
            ])
        else:
            dx = height - width
            xl = round(dx / 2)
            xr = dx - xl
            train_transform = transforms.Compose([
                transforms.Pad([xl, 0, xr, 0], fill=(255, 255, 255), padding_mode='constant'),
            ])
        image = train_transform(image)
        train_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.85278934, 0.85292006, 0.8522064], [0.311749, 0.311639, 0.31216496])])
        image = train_transform(image)

        return image, torch.tensor(self.images[item]['input_seqs']),torch.tensor(self.images[item]['output_seqs']),self.images[item]['path'],self.images[item]['label']

    def __len__(self):
        return len(self.images)
def getdataset(args):
    train_dataset = traindata(args)
    train_loader = DataLoader(train_dataset, shuffle = True, batch_size = args.batch_size, num_workers=args.num_workers,drop_last=True, pin_memory=True,)
    test_dataset = testdata(args)
    test_loader = DataLoader(test_dataset, shuffle = True, batch_size=1, num_workers=args.num_workers,drop_last=True, pin_memory=True, )
    return train_loader,test_loader