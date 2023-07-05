from src.data.make_dataset import CIFARLoader

from src.features.build_features import Flatten, SIFTBoVW, ColorHist

from src.models.RandomForest import RandomForest
from src.models.KNN import KNN
from src.models.LogisticRegression import LogisticRegression


def main():
    # Make dataset
    data_src = './data/cifar-10-batches-py/'
    data_dir = './data/preprocessed/'
    cifar_loader = CIFARLoader(data_src, data_dir)
    # Select one feature extractor in src/features/build_features.py
    data = ColorHist(data_dir=data_dir)

    X_train, y_train = data.get_train()
    X_test, y_test = data.get_test()

    print(f'Train desc shape: {X_train.shape}')
    print(f'Train labels shape: {y_train.shape}')
    print(f'Test desc shape: {X_test.shape}')
    print(f'Test labels shape: {y_test.shape}')

    model = RandomForest(save_path='./models', model_name='random_forest', n_estimators=100)
    model.fit(X_train, y_train)
    model.test(X_test, y_test)
    model.save()


if __name__ == "__main__":
    main()