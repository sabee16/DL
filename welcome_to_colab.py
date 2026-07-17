
import numpy as np
import IPython.display as display
from matplotlib import pyplot as plt
import io
import base64

ys = 200 + np.random.randn(100)
x = [x for x in range(len(ys))]

fig = plt.figure(figsize=(4, 3), facecolor='w')
plt.plot(x, ys, '-')
plt.fill_between(x, ys, 195, where=(ys > 195), facecolor='g', alpha=0.6)
plt.title("Sample Visualization", fontsize=10)

data = io.BytesIO()
plt.savefig(data)
image = F"data:image/png;base64,{base64.b64encode(data.getvalue()).decode()}"
alt = "Sample Visualization"
display.display(display.Markdown(F"""![{alt}]({image})"""))
plt.close(fig)



import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

def build_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    return model

optimizers = {
    "SGD": keras.optimizers.SGD(),
    "RMSprop": keras.optimizers.RMSprop(),
    "Adam": keras.optimizers.Adam()
}

history_dict = {}

for name, opt in optimizers.items():
    print(f"\n{'='*50}")
    print(f"Training with optimizer: {name}")
    print(f"{'='*50}")
    model = build_model()
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=5, batch_size=32,
                          validation_data=(x_test, y_test), verbose=1)
    history_dict[name] = history.history

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
for name, hist in history_dict.items():
    plt.plot(hist['loss'], label=f'{name} - Train Loss')
plt.title('Training Loss Comparison')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)

plt.subplot(1, 2, 2)
for name, hist in history_dict.items():
    plt.plot(hist['accuracy'], label=f'{name} - Train Accuracy')
plt.title('Training Accuracy Comparison')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)
plt.tight_layout(); plt.show()

print("\n" + "="*50)
print("FINAL RESULTS COMPARISON")
print("="*50)
for name, hist in history_dict.items():
    final_loss = hist['loss'][-1]
    final_acc = hist['accuracy'][-1]
    val_acc = hist['val_accuracy'][-1]
    print(f"{name:10s} | Final Train Loss: {final_loss:.4f} | "
          f"Final Train Accuracy: {final_acc:.4f} | "
          f"Validation Accuracy: {val_acc:.4f}")

best_optimizer = max(history_dict, key=lambda k: history_dict[k]['val_accuracy'][-1])
print(f"\nBest performing optimizer: {best_optimizer}")

