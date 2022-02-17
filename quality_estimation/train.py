import numpy as np
import json

from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix
from sklearn.linear_model import LogisticRegression

from quality_estimation.options import make_args
from quality_estimation.data.datasets import CVDataset
from quality_estimation.data.utils import get_statistics, scale, upsample


def main():
    args = make_args()
    dataset = CVDataset(shuffle=False, K=args.k_folds, retain_eos=args.retain_eos)
    dataset.read_data(args.path_train_src, args.path_train_mt, args.path_train_mt_info, args.path_train_labels)
    dataset.make_folds()
    accs = []
    mccs = []
    params = []
    for i in range(args.k_folds):
        indices_test = dataset.folds[i]
        indices_train = dataset.get_train_folds(i)
        model = LogisticRegression()
        X_train, y_train = dataset.collate_fn(indices_train)
        X_train, y_train = upsample(X_train, y_train)
        X_test, y_test = dataset.collate_fn(indices_test)
        mean_, scale_ = get_statistics(X_train)
        X_train = scale(X_train, mean_, scale_)
        X_test = scale(X_test, mean_, scale_)
        model.fit(X_train, y_train)
        params_i = {
            "mean_": list(mean_),
            "scale_": list(scale_),
            "coef_": list(model.coef_.squeeze()),
            "intercept_": float(model.intercept_),
        }
        if args.threshold is not None:
            predictions = model.predict_proba(X_test)
            predictions = list(map(lambda p: 1 if p[1] > args.threshold else 0, predictions))
        else:
            predictions = model.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
        acc = accuracy_score(y_test, predictions)
        mcc = matthews_corrcoef(y_test, predictions)
        params.append(params_i)
        accs.append(acc)
        mccs.append(mcc)
    if args.save_model_path is not None:
        best_params = params[np.argmax(mccs)]
        json.dump(best_params, open(args.save_model_path, "w"))


if __name__ == "__main__":
    main()
