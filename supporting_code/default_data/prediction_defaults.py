from supporting_code.enums.supported_cli_params import SupportedTrainingCLIParams
from supporting_code.default_data.training_defaults import TrainingDefaults
import os


class PredictionDefaults:
    def __init__(self):
        self.__training_defaults = TrainingDefaults(train_mode=False)
        self.__supported_cli_training_params = SupportedTrainingCLIParams()

        self.__cli_args = self.__supported_cli_training_params.get_all_args()

        # All the publicly available params.
        self.IMAGE_PATH = self.__get_image_path()
        self.CHECKPOINT_PATH = self.__get_checkpoint_path()
        self.CATEGORY_TO_JSON_FILE_PATH = self.__get_categories_to_names_json_path()
        self.TOP_K = self.__get_top_k()

    def __get_image_path(self):
        cli_img_path = self.__supported_cli_training_params.get_image_path()

        if not os.path.exists(cli_img_path):
            raise Exception('Image does not exist.')

        return cli_img_path

    def __get_checkpoint_path(self):
        cli_checkpoint_path = self.__supported_cli_training_params.get_checkpoint_path()

        return self.__training_defaults.MODEL_SAVE_PATH if cli_checkpoint_path is None else cli_checkpoint_path

    def __get_categories_to_names_json_path(self):
        cli_path = self.__cli_args[self.__supported_cli_training_params.CATEGORY_NAMES]

        return self.__training_defaults.CATEGORY_TO_JSON_FILE_NAME if cli_path is None else cli_path

    def __get_top_k(self):
        cli_top_k = self.__cli_args[self.__supported_cli_training_params.TOP_K]

        return 5 if cli_top_k is None else cli_top_k
