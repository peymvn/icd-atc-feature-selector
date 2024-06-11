"""
By: Peyman Ghasemi
Please cite as:
Ghasemi P, Lee J
Unsupervised Feature Selection to Identify Important ICD-10 and ATC Codes for Machine Learning: A Case Study on a Coronary
Artery Disease Patient Cohort
JMIR Med Inform 2024;0:e0
URL: https://medinform.jmir.org/2024/0/e0/
doi: 10.2196/52896

"""

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from concrete_autoencoder import ConcreteAutoencoderFeatureSelector
from sklearn.metrics import accuracy_score
import constants as cons
import argparse


def select_features_with_concrete_ae(
    apply_weights: bool = True,
    num_selected_features: int = 100,
    num_hidden_layers: int = 64,
    num_epochs: int = 500,
    batch_size: int = 64,
    tryout_limit: int = 1,
    start_temp: float = 20,
    min_temp: float = 0.01,
    path_to_target_variable: str = None,
    plot_training: bool = False,
):

    # get column names and weights of the one-hot-encoded df
    all_codes_df = pd.read_csv(
        os.path.join(cons.PROCESSED_DIR, "all_codes_list.csv"), na_filter=False
    )
    feature_list = all_codes_df["code"].to_list()
    class_ranks = all_codes_df["rank"].to_numpy()

    if apply_weights:
        class_weights = 1 / (class_ranks + 1)  # higher weights for lower ranks
    else:
        class_weights = np.ones_like(class_ranks)  # equal weights for all classes

    # Read and prepare the dataset
    one_hot_encoded = np.load(
        os.path.join(cons.PROCESSED_DIR, "one_hot_encoded_data.npy")
    )
    if path_to_target_variable is not None:
        target_var = pd.read_csv(path_to_target_variable)
        target_var = target_var.to_numpy()
    else:
        target_var = np.zeros((len(one_hot_encoded), 1))

    x_train, x_test, y_train, y_test = train_test_split(
        one_hot_encoded, target_var, test_size=0.33, random_state=666
    )
    # define decoder
    if path_to_target_variable is None:
        num_features = len(feature_list)

        def decoder(x):
            x = Dense(num_hidden_layers)(x)
            x = LeakyReLU(0.2)(x)
            x = Dropout(0.1)(x)
            x = Dense(num_hidden_layers)(x)
            x = LeakyReLU(0.2)(x)
            x = Dropout(0.1)(x)
            x = Dense(num_features, activation="sigmoid")(x)
            return x

    else:
        num_features = target_var.shape[1]

        def decoder(x):
            x = Dense(num_hidden_layers)(x)
            x = LeakyReLU(0.2)(x)
            x = Dropout(0.1)(x)
            x = Dense(num_hidden_layers)(x)
            x = LeakyReLU(0.2)(x)
            x = Dropout(0.1)(x)
            x = Dense(num_features, activation="sigmoid")(x)
            return x

    # Train
    selector = ConcreteAutoencoderFeatureSelector(
        K=num_selected_features,
        output_function=decoder,
        num_epochs=num_epochs,
        batch_size=batch_size,
        tryout_limit=tryout_limit,
        start_temp=start_temp,
        min_temp=min_temp,
        class_weights=class_weights,
        initial_weights=None,
        plot_training=plot_training,
    )
    # if unsupervised
    if path_to_target_variable is None:
        y_train = x_train
        y_test = x_test

    model = selector.fit(x_train, y_train, x_test, y_test)

    # Select Features
    best_feature_idx = selector.get_support(indices=True)
    best_feature_idx = np.unique(best_feature_idx)  # remove duplicates
    best_features = [feature_list[i] for i in best_feature_idx]
    best_features.sort()
    selected_features_df = all_codes_df.loc[
        all_codes_df["code"].isin(best_features),
        ["code", "chapter", "rank", "description"],
    ]
    selected_features_df.to_csv(
        os.path.join(cons.SELECTED_FEATURES_DIR, "selected_features.csv"), index=False
    )

    # Evaluate results
    print("Training Done... Evaluating the results...")
    prediction = model.predict(x_test)
    thresh = 0.5

    binary_pred = prediction.copy()
    binary_pred[binary_pred < thresh] = 0
    binary_pred[binary_pred >= thresh] = 1

    if path_to_target_variable is None:
        feature_accuracy = []
        for i in range(x_train.shape[1]):
            feature_accuracy.append(accuracy_score(x_test[:, i], binary_pred[:, i]))
        feature_accuracy_df = pd.DataFrame(
            {"Feature": feature_list, "Accuracy": feature_accuracy}
        )
        feature_accuracy_df.to_csv(
            os.path.join(cons.SELECTED_FEATURES_DIR, "feature_accuracy.csv"),
            index=False,
        )
    else:
        accuracy = accuracy_score(y_test, binary_pred)
        print(f"Accuracy: {accuracy}")

    print(feature_accuracy_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Select features using concrete autoencoder"
    )
    parser.add_argument(
        "--apply_weights",
        type=bool,
        default=True,
        help="Apply weights to the classes based on their ranks",
    )
    parser.add_argument(
        "--num_selected_features",
        type=int,
        default=100,
        help="Number of selected features",
    )
    parser.add_argument(
        "--num_hidden_layers",
        type=int,
        default=64,
        help="Number of hidden layers in the decoder",
    )
    parser.add_argument("--num_epochs", type=int, default=500, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--tryout_limit", type=int, default=1, help="Number of tryouts")
    parser.add_argument(
        "--start_temp", type=float, default=20, help="Start temperature"
    )
    parser.add_argument(
        "--min_temp", type=float, default=0.01, help="Minimum temperature"
    )
    parser.add_argument(
        "--path_to_target_variable",
        type=str,
        default=None,
        help="Path to the target variable - if you want to select features based on the target variable and not unsupervisedly",
    )

    args = parser.parse_args()
    select_features_with_concrete_ae(
        args.apply_weights,
        args.num_selected_features,
        args.num_hidden_layers,
        args.num_epochs,
        args.batch_size,
        args.tryout_limit,
        args.start_temp,
        args.min_temp,
        args.path_to_target_variable,
        plot_training=False,
    )
