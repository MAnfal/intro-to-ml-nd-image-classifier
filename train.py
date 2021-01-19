from supporting_code.default_data.training_defaults import TrainingDefaults

training_defaults = TrainingDefaults()

if __name__ == '__main__':
    print(training_defaults.get_model_save_path())
