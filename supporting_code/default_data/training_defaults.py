from supporting_code.enums.supported_training_cli_params import SupportedTrainingCLIParams
from supporting_code.enums.supported_architectures import SupportedArchitectures
import os


class TrainingDefaults:
    def __init__(self):
        # These params can be passed from the CLI.
        self.__default_save_dir = 'models'
        self.__default_architecture = SupportedArchitectures.DENSE_NET_121
        self.__default_learning_rate = 0.0008
        self.__default_hidden_units = 1024
        self.__default_epochs = 10
        self.__default_data_dir = 'flowers'

        # These params are served as defaults for now.
        self.__default_batch_size = 64
        self.__default_network_means = [0.485, 0.456, 0.406]
        self.__default_network_std_dev = [0.229, 0.224, 0.225]
        self.__default_center_crop_size = 224
        self.__default_image_resize_size = 255
        self.__default_saved_model_name = 'image_classifier_model.pt'

        self.__supported_cli_training_params = SupportedTrainingCLIParams()

        self.__cli_args = self.__supported_cli_training_params.get_all_args()
        self.__cli_data_dir = self.__supported_cli_training_params.get_data_dir()

    def get_data_directories(self):
        data_dir = self.__default_data_dir if self.__cli_data_dir is None else self.__cli_data_dir

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

    def get_model_save_path(self):
        cli_save_path = self.__cli_args[self.__supported_cli_training_params.SAVE_DIRECTORY]

        return f"{self.__default_save_dir if cli_save_path is None else cli_save_path}/{self.__default_saved_model_name}"
