from supporting_code.enums.supported_training_cli_params import SupportedTrainingCLIParams
from supporting_code.enums.supported_architectures import SupportedArchitectures
import os
import torch


class TrainingDefaults:
    def __init__(self):
        self.__supported_cli_training_params = SupportedTrainingCLIParams()
        self.__cli_args = self.__supported_cli_training_params.get_all_args()

        # All the publicly available params.
        self.DATA_DIRECTORIES = self.__get_data_directories()
        self.MODEL_SAVE_PATH = self.__get_model_save_path()
        self.NETWORK_STD_DEV = [0.229, 0.224, 0.225]
        self.NETWORK_MEANS = [0.485, 0.456, 0.406]
        self.BATCH_SIZE = 64
        self.EPOCHS = self.__get_epochs()
        self.LEARNING_RATE = self.__get_lr()
        self.HIDDEN_UNITS = self.__get_hidden_units()
        self.ARCHITECTURE = self.__get_arch()

        self.CENTER_CROP_SIZE = 224
        self.IMAGE_RESIZE_SIZE = 255

        self.CATEGORY_TO_JSON_FILE_NAME = 'cat_to_name.json'

        self.DEVICE = self.__get_device()

    def __get_data_directories(self):
        cli_data_dir = self.__supported_cli_training_params.get_data_dir()

        data_dir = 'flowers' if cli_data_dir is None else cli_data_dir

        if not os.path.exists(f"./{data_dir}"):
            raise Exception(f"Data Directory: \"{data_dir}\" does not exist.")
        else:
            data_dir_dict = {
                'train': f"{data_dir}/train",
                'test': f"{data_dir}/test",
                'valid': f"{data_dir}/valid"
            }

            for data_sub_dir in data_dir_dict.values():
                if not os.path.exists(f"./{data_sub_dir}"):
                    raise Exception(f"Data Sub Directory: \"{data_sub_dir}\" does not exist.")

        return data_dir_dict

    def __get_model_save_path(self):
        cli_save_path = self.__cli_args[self.__supported_cli_training_params.SAVE_DIRECTORY]

        return f"{'models' if cli_save_path is None else cli_save_path}/image_classifier_model.pt"

    def __get_epochs(self):
        cli_epochs = self.__cli_args[self.__supported_cli_training_params.EPOCHS]

        return 10 if cli_epochs is None else cli_epochs

    def __get_lr(self):
        cli_lr = self.__cli_args[self.__supported_cli_training_params.LEARNING_RATE]

        return 0.0008 if cli_lr is None else cli_lr

    def __get_hidden_units(self):
        cli_hu = self.__cli_args[self.__supported_cli_training_params.HIDDEN_UNITS]

        return 512 if cli_hu is None else cli_hu

    def __get_arch(self):
        cli_arch = self.__cli_args[self.__supported_cli_training_params.LEARNING_RATE]

        arch = SupportedArchitectures.DENSE_NET_121.value if cli_arch is None else cli_arch

        return arch

    def __get_device(self):
        cli_gpu = self.__cli_args[self.__supported_cli_training_params.GPU]

        is_cuda_available = torch.cuda.is_available()

        if cli_gpu and not is_cuda_available:
            raise Exception('Model can\'t run on GPU. It\'s not available.')

        return torch.device('cuda' if (cli_gpu or is_cuda_available) else 'cpu')
