
from dataset_31_march_2025 import model, new_train_ss, y_train, new_test_ss, y_test
from tensorflow.keras.callbacks import EarlyStopping
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(new_train_ss, y_train, validation_data=(new_test_ss, y_test), epochs=5, callbacks=[EarlyStopping(patience=2)])
