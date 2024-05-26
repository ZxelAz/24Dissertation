import diffml
import numpy as np
import math as m
import torch
from torch.utils.data import Dataset


class SourceImage1D:
    def __init__(self, num_pxl, img_size=1):
        """

        :param num_pxl: num_pxl in one dimension

        The unit of length is measured in lambda/NA.
        In such unit, with sigma = 0.21 (aperture for psf), Rayleigh limit d_r = 0.61.

        """

        self.num_pxl = num_pxl
        self.x = np.linspace(-img_size / 2, img_size / 2, num=num_pxl)
        self.pxl_size = self.x[1] - self.x[0]

    def generate_double_slit(self, num_img, d_min, d_max=0.61, num_photons=1):
        """

        :param num_img: total num of samples
        :param d_min: min separation
        :param d_max: max separation
        :param num_photons
        :return: The list of double slits of random separations and widths, or noise.
        Each entry contains the meshgrid of the slits and the label=1
        """

        separations = (self.pxl_size *
                       np.random.randint(low=np.floor(d_min/self.pxl_size), high=np.floor(d_max/self.pxl_size), size=num_img))

        dataset = [self.double_slit(d=separation, num_photons=num_photons)
                   for separation in separations]

        return dataset, separations

    def generate_displaced_single_slit(self, num_img, d_min, d_max=0.61, num_photons=1):
        """

        :param num_img: total num of samples
        :param d_min: min separation
        :param d_max: max separation
        :param num_photons
        :return: The list of double slits of random separations and widths, or noise.
        Each entry contains the meshgrid of the slits and the label=1
        """

        separations = (self.pxl_size *
                       np.random.randint(low=np.floor(d_min/self.pxl_size), high=np.floor(d_max/self.pxl_size), size=num_img))

        dataset = [self.displaced_single_slit(d=separation, num_photons=num_photons)
                   for separation in separations]

        
        dataset_mirror = [self.displaced_single_slit(d=-separation, num_photons=num_photons)
                   for separation in separations]

        return dataset, dataset_mirror, separations

    def double_slit(self, d, num_photons=1):
        """

        :param num_photons: the total num of photons
        :param d: the separation between slits measured in NA
        :return: an array on axis that describes the image of the double slit, each slit is one by one pxl.
        The total intensity is normalised to 1

        h is the num of photons per pxl
        """
        h = np.sqrt(num_photons / 2)

        # 1D double slit
        mask1 = (d / 2 >= self.x - self.pxl_size / 2) & (d / 2 < self.x + self.pxl_size / 2)
        mask2 = (-d / 2 >= self.x - self.pxl_size / 2) & (-d / 2 < self.x + self.pxl_size / 2)
        mask = mask1 | mask2

        source_field = h * mask

        return source_field

    def displaced_single_slit(self, d, num_photons=1):
        """

        :param num_photons: the total num of photons
        :param d: the separation between slits measured in NA
        :return: an array on axis that describes the image of the double slit, each slit is one by one pxl.
        The total intensity is normalised to 1

        h is the num of photons per pxl
        """
        h = np.sqrt(num_photons)

        # 1D double slit
        mask = (d/2 >= self.x - self.pxl_size / 2) & (d/2 < self.x + self.pxl_size / 2)
        source_field = h * mask

        return source_field


    def single_slit(self, num_photons=1):
        zero_idx = np.round(self.num_pxl / 2).astype(int)
        source_field = np.zeros(self.num_pxl)
        source_field[zero_idx] = np.sqrt(num_photons)
        return source_field


class CustomUnlabeledDataset(Dataset):
    def __init__(self, img, labels):
        self.img = img
        self.labels = labels

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        image, label = (self.img[idx], self.labels[idx])
        return torch.tensor(image, dtype=torch.complex64), torch.tensor(label, dtype=torch.float32)


def add_guassian(img, total_noise):
    '''

    Args:
        img: the batch img input
        total_noise: total guassian noise in terms of num_photons

    Returns:

    '''

    if total_noise == 0:
        return img
    else:
        num_photons_per_pxl = total_noise / img.size()[1]
        phases = torch.rand(img.size())
        noise_amplitude = torch.zeors_like(img)
        noise_amplitude = torch.normal(noise_amplitude, torch.sqrt(num_photons_per_pxl))
        noise = noise_amplitude * torch.exp(1j * phases)
        return img + noise


def normalise_img_int(img, pixel_size, intensity):
    '''

    Args:
        img: the batch img input
        pixel_size:
        intensity: the target intensity

    Returns: The img whose total intensity is 'intensity'

    '''

    original_intensity = pixel_size * torch.sum(img.abs().pow(2), dim=1).unsqueeze(1)
    normalised_img = intensity ** 0.5 * img / (original_intensity ** 0.5)
    return normalised_img


def randomize_img_int(img, pixel_size, intensity):
    '''

    Args:
        img: the batch img input
        pixel_size:
        intensity: the target intensity

    Returns: The img whose total intensity is 'intensity'

    '''
    original_intensity = pixel_size * torch.sum(img.abs().pow(2), dim=1).unsqueeze(1)
    normalised_img = intensity ** 0.5 * img / (original_intensity ** 0.5)
    return normalised_img