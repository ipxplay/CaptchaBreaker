from iitg.common import loads
from iitg import config
from iitg.core.train_model import evaluate_model, test_model, read_data_labels, prepare_data_labels

if __name__ == '__main__':
    no = 2.4
    model, lb = loads.load_model_lables(no)
    devX, devY = prepare_data_labels(*read_data_labels(config.DEV_DATA_PATH))
    evaluate_model(model, lb, 128, devX, devY)
    testAcc, testAll = test_model(model, lb, config.TEST_DATA_PATH)
    print(f'test: accuary/amount is {testAcc}/{testAll}')
    print(f'[INFO] the experiment no is {no}')
