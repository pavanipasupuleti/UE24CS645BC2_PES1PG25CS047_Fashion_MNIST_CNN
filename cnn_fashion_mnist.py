# ============================================================
# CNN FROM SCRATCH — Fashion MNIST 
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist

# ─────────────────────────────────────────────
# 1. LOAD AND PREPARE DATA
# ─────────────────────────────────────────────

print("="*60)
print("  CNN FROM SCRATCH — Fashion MNIST (OPTIMIZED)")
print("="*60)
print("\nLoading Fashion MNIST dataset...")
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Normalize: pixels are 0-255, we scale to [0, 1] to prevent gradient explosion
X_train = X_train / 255.0
X_test  = X_test  / 255.0

# Reshape to (Batch, Height, Width, Channels) for standard CNN processing
X_train = X_train.reshape(-1, 28, 28, 1)
X_test  = X_test.reshape(-1, 28, 28, 1)

# Convert integer labels (0-9) to binary vectors (e.g., 3 -> [0,0,0,1,0,0,0,0,0,0])
def one_hot(labels, num_classes=10):
    out = np.zeros((len(labels), num_classes))
    out[np.arange(len(labels)), labels] = 1
    return out

y_train_oh = one_hot(y_train)
y_test_oh  = one_hot(y_test)

# Subset selection: Training on the full 60k samples with NumPy loops would be extremely slow
TRAIN_SIZE = 5000  
TEST_SIZE  = 500    

X_small = X_train[:TRAIN_SIZE]
y_small = y_train_oh[:TRAIN_SIZE]
y_small_raw = y_train[:TRAIN_SIZE]
X_test_small = X_test[:TEST_SIZE]
y_test_small = y_test_oh[:TEST_SIZE]
y_test_small_raw = y_test[:TEST_SIZE]

# Visualization of the dataset
class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print("\nSaving sample images...")
plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_small[i].reshape(28, 28), cmap='gray')
    plt.title(class_names[y_train[i]], fontsize=8)
    plt.axis('off')
plt.suptitle("Fashion MNIST Sample Images", fontsize=12)
plt.tight_layout()
plt.savefig("sample_images.png", dpi=150, bbox_inches='tight')
plt.close()


# ─────────────────────────────────────────────
# 2. CONVOLUTION LAYER
# ─────────────────────────────────────────────

class ConvLayer:
    def __init__(self, num_filters, filter_size):
        # Initialize filters with small random values to break symmetry
        self.filters = np.random.randn(num_filters, filter_size, filter_size, 1) / 9
        self.num_filters = num_filters
        self.filter_size = filter_size

    def forward(self, x):
        """ Performs a 2D convolution (valid padding, stride 1) """
        self.input = x
        H, W, C = x.shape
        f = self.filter_size
        out_H = H - f + 1
        out_W = W - f + 1
        output = np.zeros((out_H, out_W, self.num_filters))

        # Nested loops for sliding window operation (Manual Convolution)
        for i in range(out_H):
            for j in range(out_W):
                # Extract the patch from the input image
                patch = x[i:i+f, j:j+f, :]
                for k in range(self.num_filters):
                    # Dot product of filter and patch + summation
                    output[i, j, k] = np.sum(patch * self.filters[k])
        return output

    def backward(self, d_output, lr):
        """ Calculates gradients for filters and propagates error back to input """
        f = self.filter_size
        d_filters = np.zeros_like(self.filters)
        d_input   = np.zeros_like(self.input)
        out_H, out_W, _ = d_output.shape

        for i in range(out_H):
            for j in range(out_W):
                patch = self.input[i:i+f, j:j+f, :]
                for k in range(self.num_filters):
                    # Gradient of loss w.r.t filters: sum of (patch * incoming error)
                    d_filters[k] += d_output[i, j, k] * patch
                    # Gradient of loss w.r.t input: sum of (filter * incoming error)
                    d_input[i:i+f, j:j+f, :] += d_output[i, j, k] * self.filters[k]

        # Update weights using SGD
        self.filters -= lr * d_filters
        return d_input


# ─────────────────────────────────────────────
# 3. RELU ACTIVATION
# ─────────────────────────────────────────────

class ReLU:
    def forward(self, x):
        self.input = x
        # Activation function: f(x) = max(0, x)
        return np.maximum(0, x)

    def backward(self, d_output):
        # Gradient is 1 if input > 0, else 0
        d = d_output.copy()
        d[self.input <= 0] = 0
        return d


# ─────────────────────────────────────────────
# 4. MAXPOOL LAYER
# ─────────────────────────────────────────────

class MaxPool:
    def __init__(self, pool_size=2):
        self.p = pool_size

    def forward(self, x):
        """ Downsamples the image by taking the maximum value in every p x p patch """
        self.input = x
        H, W, C = x.shape
        p = self.p
        out_H, out_W = H // p, W // p
        output = np.zeros((out_H, out_W, C))

        for i in range(out_H):
            for j in range(out_W):
                region = x[i*p:(i+1)*p, j*p:(j+1)*p, :]
                output[i, j, :] = np.max(region, axis=(0, 1))
        return output

    def backward(self, d_output):
        """ Routes the gradient only to the neuron that had the maximum value during forward pass """
        p = self.p
        d_input = np.zeros_like(self.input)
        out_H, out_W, C = d_output.shape

        for i in range(out_H):
            for j in range(out_W):
                region = self.input[i*p:(i+1)*p, j*p:(j+1)*p, :]
                for c in range(C):
                    # Identify the index of the max value
                    max_val = np.max(region[:, :, c])
                    mask = (region[:, :, c] == max_val)
                    # Pass the gradient only to that specific index
                    d_input[i*p:(i+1)*p, j*p:(j+1)*p, c] += mask * d_output[i, j, c]
        return d_input


# ─────────────────────────────────────────────
# 5. FLATTEN
# ─────────────────────────────────────────────

class Flatten:
    def forward(self, x):
        # Store shape to allow reshaping back during backprop
        self.shape = x.shape
        # Convert 3D volume to 1D vector
        return x.flatten()

    def backward(self, d):
        # Reshape the 1D gradient back to the 3D volume shape
        return d.reshape(self.shape)


# ─────────────────────────────────────────────
# 6. FULLY CONNECTED LAYER
# ─────────────────────────────────────────────

class FullyConnected:
    def __init__(self, input_size, output_size):
        # He Initialization: helpful for layers following ReLU
        self.W = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.b = np.zeros(output_size)

    def softmax(self, x):
        """ Converts raw scores (logits) into probabilities """
        e = np.exp(x - np.max(x)) # Numerical stability subtraction
        return e / e.sum()

    def forward(self, x):
        self.input = x
        # Y = Wx + b
        return self.softmax(np.dot(x, self.W) + self.b)

    def backward(self, d_output, lr):
        """ Gradients for weight, bias, and input """
        # Weight gradient: outer product of input and error
        d_W = np.outer(self.input, d_output)
        d_b = d_output
        # Propagate error back to the previous layer
        d_input = np.dot(self.W, d_output)
        
        # Update weights and biases
        self.W -= lr * d_W
        self.b -= lr * d_b
        return d_input


# ─────────────────────────────────────────────
# 7. LOSS FUNCTIONS
# ─────────────────────────────────────────────

def cross_entropy_loss(pred, actual):
    # Measures how far the prediction is from the truth (target)
    return -np.sum(actual * np.log(pred + 1e-10))

def loss_gradient(pred, actual):
    # Simplification of Softmax + CrossEntropy derivative
    return pred - actual


# ─────────────────────────────────────────────
# 8. TRAINING LOOP
# ─────────────────────────────────────────────

def train(X, y, epochs=5, lr=0.01, num_filters=16):
    print("\n" + "="*60)
    print(f"  TRAINING STARTED ({epochs} epochs, {num_filters} filters, lr={lr})")
    print("="*60)
    
    # Instantiate layers
    conv    = ConvLayer(num_filters=num_filters, filter_size=3) 
    relu    = ReLU()
    pool    = MaxPool(pool_size=2)
    flat    = Flatten()
    # Input size: (28-3+1)/2 = 13. Area = 13*13. Volume = 13*13*num_filters.
    fc      = FullyConnected(input_size=13*13*num_filters, output_size=10)

    loss_history = []
    acc_history  = []
    epoch_details = []  # Store detailed epoch information

    for epoch in range(epochs):
        total_loss = 0
        correct    = 0

        for i in range(len(X)):
            # --- FORWARD PASS ---
            out = conv.forward(X[i])
            out = relu.forward(out)
            out = pool.forward(out)
            out = flat.forward(out)
            pred = fc.forward(out)

            # Calculation of metrics
            loss = cross_entropy_loss(pred, y[i])
            total_loss += loss
            
            if np.argmax(pred) == np.argmax(y[i]):
                correct += 1

            # --- BACKWARD PASS (The order is reversed) ---
            grad = loss_gradient(pred, y[i])
            grad = fc.backward(grad, lr)
            grad = flat.backward(grad)
            grad = pool.backward(grad)
            grad = relu.backward(grad)
            conv.backward(grad, lr)

            # Logging
            if (i + 1) % 500 == 0:
                running_acc = (correct / (i+1)) * 100
                running_loss = total_loss / (i+1)
                print(f"  Epoch {epoch+1} | Sample {i+1}/{len(X)} | "
                      f"Loss: {running_loss:.4f} | Acc: {running_acc:.2f}%")

        avg_loss = total_loss / len(X)
        acc      = (correct / len(X)) * 100
        loss_history.append(avg_loss)
        acc_history.append(acc)
        
        # Store epoch details for later display
        epoch_details.append({
            'epoch': epoch + 1,
            'loss': avg_loss,
            'accuracy': acc
        })
        
        print(f"\n{'─'*60}")
        print(f" EPOCH {epoch+1} COMPLETE")
        print(f"   Loss     : {avg_loss:.4f}")
        print(f"   Accuracy : {acc:.2f}%")
        print(f"{'─'*60}\n")

    return conv, relu, pool, flat, fc, loss_history, acc_history, epoch_details


# ─────────────────────────────────────────────
# 9. EVALUATION
# ─────────────────────────────────────────────

def evaluate(X, y, conv, relu, pool, flat, fc):
    """ Runs a forward pass on unseen data without updating weights """
    print("\n" + "="*60)
    print("  EVALUATING ON TEST SET")
    print("="*60)
    
    correct = 0
    for i in range(len(X)):
        out  = conv.forward(X[i])
        out  = relu.forward(out)
        out  = pool.forward(out)
        out  = flat.forward(out)
        pred = fc.forward(out)
        if np.argmax(pred) == np.argmax(y[i]):
            correct += 1
    
    acc = (correct / len(X)) * 100
    print(f"\n  Test Accuracy: {acc:.2f}%")
    return acc


# ─────────────────────────────────────────────
# 10. DISPLAY ALL EPOCHS SUMMARY
# ─────────────────────────────────────────────

def display_all_epochs_summary(epoch_details):
    """ Display all epochs together after training completion """
    print("\n" + "="*60)
    print("  TRAINING SUMMARY - ALL EPOCHS")
    print("="*60)
    
    for detail in epoch_details:
        print(f"\n EPOCH {detail['epoch']} COMPLETE")
        print(f"   Loss     : {detail['loss']:.4f}")
        print(f"   Accuracy : {detail['accuracy']:.2f}%")
        print(f" {'─'*56}")  # Separator line for feature output
    
    print("\n" + "="*60)


# ─────────────────────────────────────────────
# 11. VISUALIZATION - SEPARATE GRAPHS
# ─────────────────────────────────────────────

def plot_epoch_vs_loss(loss_history):
    """ Plots epoch vs. loss progression  """
    epochs = list(range(1, len(loss_history) + 1))
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot Loss
    color = '#e74c3c'
    ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=13, fontweight='bold', color=color)
    ax.plot(epochs, loss_history, 's-', color=color, linewidth=3.5, markersize=8, label='Training Loss')
    ax.tick_params(axis='y', labelcolor=color)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.title('Training Progress: Epoch vs Loss', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig('epoch_vs_loss.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n✓ Graph saved: epoch_vs_loss.png")


def plot_epoch_vs_accuracy(acc_history):
    """ Plots epoch vs. accuracy progression """
    epochs = list(range(1, len(acc_history) + 1))
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot Accuracy
    color = '#27ae60'
    ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold', color=color)
    ax.plot(epochs, acc_history, 'o-', color=color, linewidth=3.5, markersize=8, label='Training Accuracy')
    ax.tick_params(axis='y', labelcolor=color)
    ax.set_ylim([0, 100])
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.title('Training Progress: Epoch vs Accuracy', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig('epoch_vs_accuracy.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Graph saved: epoch_vs_accuracy.png")


def plot_combined_metrics(loss_history, acc_history):
    """ Plots both loss and accuracy on same figure with separate subplots """
    epochs = list(range(1, len(acc_history) + 1))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot Loss (Left Subplot)
    color1 = '#e74c3c'
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold', color=color1)
    ax1.plot(epochs, loss_history, 's-', color=color1, linewidth=3.5, markersize=8)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_title('Epoch vs Loss', fontsize=13, fontweight='bold')
    
    # Plot Accuracy (Right Subplot)
    color2 = '#27ae60'
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold', color=color2)
    ax2.plot(epochs, acc_history, 'o-', color=color2, linewidth=3.5, markersize=8)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim([0, 100])
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_title('Epoch vs Accuracy', fontsize=13, fontweight='bold')
    
    plt.suptitle('Training Progress: Loss and Accuracy Over Epochs', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('training_progress_combined.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Graph saved: training_progress_combined.png")


# ─────────────────────────────────────────────
# 12. EXECUTION
# ─────────────────────────────────────────────

conv, relu, pool, flat, fc, loss_hist, acc_hist, epoch_details = train(
    X_small, y_small, 
    epochs=5,        
    lr=0.01,         
    num_filters=16   
)

test_acc = evaluate(X_test_small, y_test_small, conv, relu, pool, flat, fc)

# Display all epochs summary with separator lines
display_all_epochs_summary(epoch_details)

# Generate all three visualization options
print("\n" + "="*60)
print("  GENERATING VISUALIZATIONS")
print("="*60)

plot_epoch_vs_loss(loss_hist)
plot_epoch_vs_accuracy(acc_hist)
plot_combined_metrics(loss_hist, acc_hist)

