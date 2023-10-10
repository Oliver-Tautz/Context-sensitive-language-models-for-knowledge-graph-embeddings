import numpy as np

def merge_historires(history_list):
    h = {}
    for key in history_list[0].history.keys():
        h[key] = [h.history[key][0] for h in history_list]
    return h


def get_vectors(triples, entity_vec_mapping, vector_size=200):
    X = np.array(triples)
    X = [(entity_vec_mapping(x[0]), entity_vec_mapping(x[1]), entity_vec_mapping(x[2])) for x in X]
    X = [np.concatenate(x) for x in X]
    X = np.vstack(X).astype(np.float64)

    return X


def test_sklearn_model(model, X, Y, x_test, y_test, subset=10000):
    ix = np.random.choice(range(len(X)), size=subset)

    scaler = preprocessing.StandardScaler().fit(X)

    X_scaled = scaler.transform(X[ix])
    model.fit(X_scaled, Y[ix])

    print(f'train_score ={model.score(scaler.transform(X), Y)}')
    print(f'test_score ={model.score(scaler.transform(x_test), y_test)}')


def scale_and_predict(model, x):
    x = preprocessing.StandardScaler().fit_transform(x)
    return model.predict(x)


def fast_concat(se, pe, oe):
    assert se.shape == pe.shape, "Error! fast_concat with differing shapes"
    assert se.shape == oe.shape, "Error! fast_concat with differing shapes"

    x = np.empty((se.shape[0], se.shape[1] * 3), dtype=np.float32)
    x[:, 0:100] = se
    x[:, 100:200] = pe
    x[:, 200:] = oe

    return x