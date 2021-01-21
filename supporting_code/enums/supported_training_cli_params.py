from argparse import ArgumentParser


class SupportedTrainingCLIParams:
    def __init__(self):
        self.SAVE_DIRECTORY = 'save_dir'
        self.ARCHITECTURE = 'arch'
        self.LEARNING_RATE = 'learning_rate'
        self.HIDDEN_UNITS = 'hidden_units'
        self.EPOCHS = 'epochs'
        self.GPU = 'gpu'

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
        
        self.__parsed_args = self.__arg_parser.parse_known_args()

        self.__other_args = vars(self.__parsed_args[0])

        self.__data_dir = self.__get_parsed_data_dir()

    def get_all_args(self):
        return self.__other_args

    def get_data_dir(self):
        return self.__data_dir

    def __get_parsed_data_dir(self):
        parsed_data_dir = self.__parsed_args[1]
        data_dir = None

        if len(parsed_data_dir) > 0:
            data_dir = parsed_data_dir[0]

        return data_dir
