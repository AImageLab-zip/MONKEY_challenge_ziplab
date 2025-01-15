import torch

import models


class ModelFactory:
    @staticmethod
    def get_model(model_name, **kwargs):
        name = model_name.lower()  # get model name from config, if not found use fallback_model, convert to lowercase
        model_class = None

        # List of custom model names excluding internal attributes
        custom_model_names = [
            model for model in models.__dict__ if not model.startswith("__")
        ]

        # Check for partial matches in the custom models
        for key in custom_model_names:
            if name in key.lower() or key.lower() in name:
                model_class = getattr(models, key)
                break

        if model_class is None:
            # Check for partial matches in the torch.nn.modules models if no match found in custom models
            torch_model_names = [
                model
                for model in torch.nn.modules.__dict__
                if not model.startswith("__")
            ]
            for key in torch_model_names:
                if name in key.lower() or key.lower() in name:
                    model_class = getattr(torch.nn.modules, key)
                    break

        if model_class:
            try:
                # Pass **kwargs along with args and config to the model constructor
                model_instance = model_class(**kwargs)
            except TypeError as e:
                raise TypeError(
                    f"Could not instantiate {model_class} with {kwargs}: {e}"
                )
        else:
            raise Exception(f"Model {name} not implemented")

        return model_instance
