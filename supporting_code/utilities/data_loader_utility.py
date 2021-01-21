from torchvision import datasets, transforms, models
import torch


class DataLoaderUtility:
    def get_data_loaders(
            self,
            dir_dict,
            center_crop_size,
            image_resize_size,
            network_means,
            network_std_dev,
            batch_size
    ):
        train_dir = dir_dict['train']
        test_dir = dir_dict['test']
        valid_dir = dir_dict['valid']

        train_transform = self.__get_train_transform(center_crop_size, network_means, network_std_dev)
        test_transform = self.__get_test_transform(image_resize_size, center_crop_size, network_means, network_std_dev)

        train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
        test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
        validation_dataset = datasets.ImageFolder(valid_dir, transform=test_transform)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )

        validation_loader = torch.utils.data.DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )

        return {
            'train': (train_dataset, train_loader),
            'test': (test_dataset, test_loader),
            'valid': (validation_dataset, validation_loader)
        }

    def __get_train_transform(self, center_crop_size, network_means, network_std_dev):
        return transforms.Compose(
            [
                transforms.RandomRotation(30),
                transforms.RandomResizedCrop(center_crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(network_means, network_std_dev)
            ]
        )

    def __get_test_transform(self, image_resize_size, center_crop_size, network_means, network_std_dev):
        return transforms.Compose(
            [
                transforms.Resize(image_resize_size),
                transforms.CenterCrop(center_crop_size),
                transforms.ToTensor(),
                transforms.Normalize(network_means, network_std_dev)
            ]
        )
