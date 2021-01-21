from supporting_code.default_data.prediction_defaults import PredictionDefaults
from supporting_code.default_data.training_defaults import TrainingDefaults
from supporting_code.utilities.image_utilities import get_image_as_np_array
import json
import torch

prediction_defaults = PredictionDefaults()
training_defaults = TrainingDefaults(train_mode=False)


def get_saved_model():
    model = torch.load(prediction_defaults.CHECKPOINT_PATH)

    return model.to(training_defaults.DEVICE)


def get_np_image():
    return get_image_as_np_array(
        prediction_defaults.IMAGE_PATH,
        training_defaults.IMAGE_RESIZE_SIZE,
        training_defaults.CENTER_CROP_SIZE,
        training_defaults.NETWORK_MEANS,
        training_defaults.NETWORK_STD_DEV
    )


''' 
Predict the class (or classes) of an image using a trained deep learning model.
'''


def predict(model):
    with torch.no_grad():
        test_image = get_np_image()

        test_image = torch.from_numpy(test_image).float().unsqueeze_(0)

        test_image = test_image.to(training_defaults.DEVICE)

        test_output = model.forward(test_image)

        test_p = torch.exp(test_output)

        return test_p.topk(prediction_defaults.TOP_K, dim=1)


def get_index_to_class_map(model):
    class_to_idx = model.class_to_idx

    return dict((v, k) for k, v in class_to_idx.items())


def get_cat_to_name_map():
    with open(prediction_defaults.CATEGORY_TO_JSON_FILE_PATH, 'r') as f:
        cat_to_name_map = json.load(f)

    return cat_to_name_map


def get_top_class_names(cat_to_name, idx_to_class, top_classes):
    return [cat_to_name[idx_to_class[top_class]] for top_class in top_classes[0].tolist()]


if __name__ == '__main__':
    loaded_model = get_saved_model()

    loaded_index_to_class = get_index_to_class_map(loaded_model)

    loaded_cat_to_name_map = get_cat_to_name_map()

    _, top_predicted_classes = predict(loaded_model)

    top_class_list = get_top_class_names(loaded_cat_to_name_map, loaded_index_to_class, top_predicted_classes)

    print(f"Top predicted classes are: {top_class_list}")
