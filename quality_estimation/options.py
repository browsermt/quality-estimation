from argparse import ArgumentParser


def make_args():
    parser = ArgumentParser()
    parser.add_argument("--path_train_src", required=True)
    parser.add_argument("--path_train_mt", required=True)
    parser.add_argument("--path_train_mt_info", required=True)
    parser.add_argument("--path_train_labels", required=True)
    parser.add_argument("--k_folds", default=5, type=int)
    parser.add_argument("--save_model_path", default=None, type=str, required=False)
    parser.add_argument("--retain_eos", action="store_true", default=False)
    parser.add_argument("--threshold", default=None, type=float, required=False)
    args = parser.parse_args()
    return args
