from argparse import ArgumentParser


class SupportedTrainingCLIParams:
    def __init__(self):
        self.SAVE_DIRECTORY = 'save_dir'
        self.ARCHITECTURE = 'arch'
        self.LEARNING_RATE = 'learning_rate'
        self.HIDDEN_UNITS = 'hidden_units'
        self.EPOCHS = 'epochs'
        self.GPU = 'gpu'
        self.CATEGORY_NAMES = 'category_names'
        self.TOP_K = 'top_k'

        self.__arg_parser = ArgumentParser()

        self.__arg_parser.add_argument(
            '--' + self.SAVE_DIRECTORY,
            help='Directory location to save the model.'
        )

        self.__arg_parser.add_argument(
            '--' + self.ARCHITECTURE,
            help='Model architecture to use during transfer learning.'
        )

        self.__arg_parser.add_argument(
            '--' + self.LEARNING_RATE,
            type=float,
            help='Learning rate at which the model should train.'
        )

        self.__arg_parser.add_argument(
            '--' + self.HIDDEN_UNITS,
            type=int,
            help='Number of hidden units for the hidden layer.'
        )

        self.__arg_parser.add_argument(
            '--' + self.EPOCHS,
            type=float,
            help='The number of times a model should train for.'
        )

        self.__arg_parser.add_argument(
            '--' + self.GPU,
            action='store_true',
            default=False,
            help='Force the model on GPU for training.'
        )

        self.__arg_parser.add_argument(
            '--' + self.CATEGORY_NAMES,
            help='Path to JSON to use while mapping categories to names.'
        )

        self.__arg_parser.add_argument(
            '--' + self.TOP_K,
            type=int,
            help='Number of top predictions to return.'
        )
        
        self.__parsed_args = self.__arg_parser.parse_known_args()

        self.__other_args = vars(self.__parsed_args[0])

        self.__data_dir = self.__get_parsed_data_dir()

    def get_all_args(self):
        return self.__other_args

    def get_data_dir(self):
        return self.__data_dir

    def __get_anonymous_parsed_args(self):
        return self.__parsed_args[1]

    def __get_parsed_data_dir(self):
        anon_parsed_args = self.__get_anonymous_parsed_args()

        if len(anon_parsed_args) < 1:
            data_dir = None
        else:
            data_dir = anon_parsed_args[0]

        return data_dir

    def get_image_path(self):
        anon_parsed_args = self.__get_anonymous_parsed_args()

        if len(anon_parsed_args) < 1:
            raise Exception('Image path required.')

        return anon_parsed_args[0]

    def get_checkpoint_path(self):
        anon_parsed_args = self.__get_anonymous_parsed_args()

        if len(anon_parsed_args) < 2:
            checkpoint_path = None
        else:
            checkpoint_path = anon_parsed_args[1]

        return checkpoint_path
