from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler
from qc_time_estimator.metrics import percentile_rel_90
from keras.optimizers import Adam


def get_NN_model(input_dim=22, nodes_per_layer=[10], dropout=0.1,
                 optimizer='adam', loss='mean_absolute_percentage_error',
                 learning_rate=0.001):

    # create Keras NN model
    model = Sequential()

    # hidden layers
    for i, n_nodes in enumerate(nodes_per_layer):
        model.add(Dense(n_nodes,
                        input_dim=input_dim if i==0 else None,
                        kernel_initializer='normal',
                        activation='relu'))
        if dropout:
            model.add(Dropout(rate=dropout))

    # output layer
    model.add(Dense(1,
                    kernel_initializer='normal',
                    activation='linear', # the default
                    ))

    if optimizer == 'adam':
        optimizer = Adam(learning_rate=learning_rate)

    # Compile model
    model.compile(optimizer=optimizer, loss=loss, metrics=['mse'])

    return model


# NN model steps, must standardize
nn_model_steps = {
    'standardize': StandardScaler(),
    'nn_model': KerasRegressor(build_fn=get_NN_model,
                               epochs=100,
                               batch_size=128,
                               verbose=0)
}
