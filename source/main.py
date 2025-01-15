from experiments.BaselineDetectronExperiment import BaselineDetectronExperiment
from utils.config_parser import get_args_and_config


def test_torch():
    """
    Test if torch is installed and working with GPU
    """
    import torch

    print(torch.__version__)
    print(torch.cuda.is_available())
    print("It's working!")


if __name__ == "__main__":
    # test_torch()

    # Path to ASAP binaries
    # asap_bin_path = "/opt/ASAP/bin"

    # Append to system PATH
    # os.environ["PATH"] += os.pathsep + asap_bin_path
    # print("ASAP Path added to PATH:", asap_bin_path)

    # asap_python_lib_path = "/opt/ASAP/lib/python3.8/site-packages"
    # sys.path.append(asap_python_lib_path)

    # Verify it's in sys.path
    # print("ASAP Python library path added:", asap_python_lib_path)

    # print(os.getcwd())
    args, config = get_args_and_config()
    # print(config)
    # data_preparator = DataPreparator(config)
    # dataset_df, folds_paths_dict = data_preparator.prepare_data()
    # print(dataset_df.head())
    # print(folds_paths_dict)
    experiment = BaselineDetectronExperiment(args, config)
    experiment.train()
    # experiment.test()
