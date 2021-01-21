from supporting_code.default_data.training_defaults import TrainingDefaults
import json
import torch
from torchvision import models
from torch import nn, optim
from collections import OrderedDict
from supporting_code.utilities.data_loader_utility import DataLoaderUtility

'''
Important initializations.
'''


def get_data_loader_dict():
    data_loader_utility = DataLoaderUtility()

    return data_loader_utility.get_data_loaders(
        training_defaults.DATA_DIRECTORIES,
        training_defaults.CENTER_CROP_SIZE,
        training_defaults.IMAGE_RESIZE_SIZE,
        training_defaults.NETWORK_MEANS,
        training_defaults.NETWORK_STD_DEV,
        training_defaults.BATCH_SIZE
    )


training_defaults = TrainingDefaults()

data_dict = get_data_loader_dict()

with open(training_defaults.CATEGORY_TO_JSON_FILE_NAME, 'r') as f:
    CATEGORY_TO_NAME_MAP = json.load(f)

'''
Training code.
'''
def get_new_classifier():
    return nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(1024, training_defaults.HIDDEN_UNITS)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(p=0.2)),

        ('fc2', nn.Linear(training_defaults.HIDDEN_UNITS, 256)),
        ('relu2', nn.ReLU()),
        ('dropout2', nn.Dropout(p=0.2)),

        ('fc3', nn.Linear(256, len(CATEGORY_TO_NAME_MAP))),
        ('output', nn.LogSoftmax(dim=1))
    ]))


def get_model_for_training():
    model = eval(f"models.{training_defaults.ARCHITECTURE}(pretrained=True)")

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = get_new_classifier()

    return model.to(training_defaults.DEVICE)


def get_criterion():
    return nn.NLLLoss()


def get_optimizer(model):
    return optim.Adam(model.classifier.parameters(), lr=training_defaults.LEARNING_RATE)


def get_trained_model():
    model = get_model_for_training()

    criterion = get_criterion()

    optimizer = get_optimizer(model)

    previous_accuracy = -1

    train_dataset, train_loader = data_dict['train']
    validation_dataset, validation_loader = data_dict['valid']

    # Train the model.
    for epoch in range(training_defaults.EPOCHS):
        train_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(training_defaults.DEVICE), labels.to(training_defaults.DEVICE)

            outputs = model.forward(images)

            optimizer.zero_grad()

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

        else:
            model.eval()

            with torch.no_grad():
                validation_loss = 0

                current_accuracy = 0

                for validation_images, validation_labels in validation_loader:
                    validation_images, validation_labels = validation_images.to(training_defaults.DEVICE), \
                                                           validation_labels.to(training_defaults.DEVICE)

                    validation_outputs = model.forward(validation_images)

                    validation_loss = criterion(validation_outputs, validation_labels)

                    validation_loss += validation_loss.item()

                    validation_ps = torch.exp(validation_outputs)

                    validation_top_p, validation_top_class = validation_ps.topk(1, dim=1)

                    validation_equal_result = validation_top_class == validation_labels.view(
                        *validation_top_class.shape)

                    current_accuracy += torch.mean(validation_equal_result.type(torch.FloatTensor))

                calc_current_accuracy = current_accuracy / len(validation_loader)

                if previous_accuracy != -1 and calc_current_accuracy < previous_accuracy:
                    print(
                        f"\n\nCurrent accuracy: {calc_current_accuracy:.3f} < "
                        f"Previous accuracy: {previous_accuracy:.3f}. "
                        f"Stopping early for the best model."
                    )

                    # We are breaking out prematurely. Let's make sure we return the model to one state.
                    model.train()

                    break
                else:
                    calc_train_loss = train_loss / len(train_loader)
                    calc_validation_loss = validation_loss / len(validation_loader)

                    print(f"Epoch {epoch + 1}/{training_defaults.EPOCHS} | "
                          f"Train loss: {calc_train_loss:.3f} | "
                          f"Validation loss: {calc_validation_loss:.3f} | "
                          f"Validation accuracy: {calc_current_accuracy:.3f}")

                    previous_accuracy = calc_current_accuracy

            model.train()

    print(f"\n\nModel training complete with the final accuracy of: {previous_accuracy * 100:.2f}%.")

    return model


def test_trained_model(model):
    test_loss = 0
    test_accuracy = 0

    test_dataset, test_loader = data_dict['test']

    criterion = get_criterion()

    model.eval()

    with torch.no_grad():
        for test_images, test_labels in test_loader:
            test_images, test_labels = test_images.to(training_defaults.DEVICE), test_labels.to(training_defaults.DEVICE)

            test_outputs = model.forward(test_images)

            test_loss = criterion(test_outputs, test_labels)

            test_loss += test_loss.item()

            test_ps = torch.exp(test_outputs)

            test_top_p, test_top_class = test_ps.topk(1, dim=1)

            test_equal_result = test_top_class == test_labels.view(*test_top_class.shape)

            test_accuracy += torch.mean(test_equal_result.type(torch.FloatTensor))

    model.train()

    calc_test_accuracy = test_accuracy / len(test_loader)
    calc_test_loss = test_loss / len(test_loader)

    print(f"Test loss: {calc_test_loss:.3f} | "
          f"Test accuracy: {calc_test_accuracy:.3f}")


def save_trained_model(model):
    model.eval()

    train_dataset, _ = data_dict['train']

    optimizer = get_optimizer(model)

    model.class_to_idx = train_dataset.class_to_idx

    model.optimizer_state_dict = optimizer.state_dict

    torch.save(model, training_defaults.MODEL_SAVE_PATH)


if __name__ == '__main__':
    trained_model = get_trained_model()

    test_trained_model(trained_model)

    save_trained_model(trained_model)
