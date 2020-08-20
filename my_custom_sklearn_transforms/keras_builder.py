from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from tensorflow.keras.layers import deserialize, serialize
from tensorflow.python.keras.saving import saving_utils

def unpack(model, training_config, weights):
    restored_model = deserialize(model)
    if training_config is not None:
        restored_model.compile(
            **saving_utils.compile_args_from_training_config(
                training_config
            )
        )
    restored_model.set_weights(weights)
    return restored_model

# Hotfix function
def make_keras_picklable():
    def __reduce__(self):
        model_metadata = saving_utils.model_metadata(self)
        training_config = model_metadata.get("training_config", None)
        model = serialize(self)
        weights = self.get_weights()
        return (unpack, (model, training_config, weights))

    cls = Model
    cls.__reduce__ = __reduce__

# Run the function
make_keras_picklable()

def createKerasModel():
    model = Sequential()
    model.add(Dense(100,
                    input_dim=12,
                    kernel_initializer=glorot_uniform(seed=4444),  
                    activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(100, 
                    kernel_initializer=glorot_uniform(seed=4444), 
                    activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(6, 
                    kernel_initializer=glorot_uniform(seed=4444), 
                    activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', 
                  optimizer='adam', 
                  metrics=['accuracy'])
    return model
    
class KerasBuilder():
    def KerasClassificator(self, epochs=120, batch_size=32, verbose=1) : 
        return KerasClassifier(build_fn=self.createKerasModel, epochs=epochs,  batch_size=batch_size, verbose=verbose)
