from src.utils import *


def process(inPath, outPath):
    # read input file, assuming col 0 is index and col -1 is target
    files, train, predict = load_input(inPath)
    X_train, X_test, y_train, y_test = perform_undersampling(train)
    trained_models, voting_model = train_models(X_train, X_test, y_train, y_test)
    make_prediction(predict, outPath, trained_models, voting_model)


if __name__ == '__main__':
    process("input.csv", "results.csv")
