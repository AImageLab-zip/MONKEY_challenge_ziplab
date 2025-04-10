import os

def get_classifier(version: str):
    from pathlib import Path
    from train import PatchClassifier_pl
    from Classifiers import PatchClassifier
    import torch

    path_to_chkpt = Path(os.environ["INSTANSEG_OUTPUT_PATH"]) / f"patch_classifier/{version}/checkpoints"
    chkpt = os.listdir(path_to_chkpt)[0]
    chkpt_path = path_to_chkpt / chkpt

    model = PatchClassifier_pl.load_from_checkpoint(
                                checkpoint_path=chkpt_path,
                                strict = True)

    # output_path = Path(path_to_chkpt).parents[0]/"model_weights.pth"

    # torch.save({'model_state_dict': model.path_classifier.state_dict(),}, output_path)

    # classifier = PatchClassifier(num_classes = model.num_classes,
    #                         patch_size = model.patch_size,
    #                         dim_in = int(model.dim_in),
    #                         from_file = output_path,
    #                         )

    return model

