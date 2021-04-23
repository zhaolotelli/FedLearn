import pandas as pd

def preprocess():
    train_data = pd.read_csv("datasets/adult/adult.data", sep = ', ', header=None, names = ('age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 
        'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 
        'hours-per-week', 'native-country', '>50K'), na_values = '?')
    test_data = pd.read_csv("datasets/adult/adult.test", sep = ', ', header=None, names = ('age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 
        'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 
        'hours-per-week', 'native-country', '>50K'), na_values = '?', skiprows = 1)

    cont_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    #cato_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
    label_col = ['>50K']

    cato_train_data = train_data.drop(cont_cols, axis = 1)
    X_train, y_train = cato_train_data.drop(label_col, axis = 1), cato_train_data[label_col]
    cato_test_data = test_data.drop(cont_cols, axis = 1)
    X_test, y_test = cato_test_data.drop(label_col, axis = 1), cato_test_data[label_col]

    X_total = pd.concat([X_train, X_test], axis = 0)
    N = X_train.shape[0]

    OH_total = pd.get_dummies(X_total)
    OH_train = OH_total[:N]
    OH_test = OH_total[N:]

    new_X_train = OH_train.to_numpy()
    new_X_test = OH_test.to_numpy()

    new_y_train = y_train['>50K'].map({'<=50K': 0, '>50K': 1}).astype(int).to_numpy()
    new_y_test = y_test['>50K'].map({'<=50K.': 0, '>50K.': 1}).astype(int).to_numpy()
    
    return (new_X_train, new_y_train), (new_X_test, new_y_test)