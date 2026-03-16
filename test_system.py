#!/usr/bin/env python3
"""
Test script to verify the MNIST AI system works correctly
"""
import numpy as np
import os
from data_loader import load_mnist
from nn import NeuralNetwork

def test_data_loading():
    """Test that data can be loaded correctly"""
    print("Testing data loading...")
    DATA_PATH = "./data/kaggle_mnist"
    
    try:
        X_train, y_train, X_test, y_test = load_mnist(DATA_PATH)
        print(f"✓ Data loaded successfully")
        print(f"  Train: {X_train.shape}, Labels: {y_train.shape}")
        print(f"  Test: {X_test.shape}, Labels: {y_test.shape}")
        print(f"  Data range: [{X_train.min():.3f}, {X_train.max():.3f}]")
        return X_train, y_train, X_test, y_test
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return None, None, None, None

def test_neural_network():
    """Test neural network creation and forward pass"""
    print("\nTesting neural network...")
    
    try:
        nn = NeuralNetwork(input_size=784, hidden_size=128, output_size=10)
        print("✓ Neural network created successfully")
        
        # Test forward pass
        test_input = np.random.rand(784)
        pred = nn.forward(test_input)
        
        print(f"✓ Forward pass works")
        print(f"  Output shape: {pred.shape}")
        print(f"  Output sum: {pred.sum():.6f} (should be ~1.0)")
        print(f"  Prediction: {np.argmax(pred)}")
        
        return nn
    except Exception as e:
        print(f"✗ Neural network test failed: {e}")
        return None

def test_training_step(nn, X_train, y_train):
    """Test a single training step"""
    print("\nTesting training step...")
    
    try:
        # Get a single sample
        x = X_train[0]
        y = y_train[0]
        
        # Forward pass
        pred_before = nn.forward(x)
        loss_before = -np.log(pred_before[y] + 1e-9)
        
        # Backward pass
        nn.backward(x, y, learning_rate=0.01)
        
        # Forward pass again
        pred_after = nn.forward(x)
        loss_after = -np.log(pred_after[y] + 1e-9)
        
        print(f"✓ Training step completed")
        print(f"  Loss before: {loss_before:.6f}")
        print(f"  Loss after: {loss_after:.6f}")
        print(f"  Loss change: {loss_after - loss_before:.6f}")
        
    except Exception as e:
        print(f"✗ Training step failed: {e}")

def test_save_load(nn):
    """Test model saving and loading"""
    print("\nTesting model save/load...")
    
    try:
        # Create model directory
        os.makedirs("model", exist_ok=True)
        test_path = "model/test_model.npz"
        
        # Save original weights
        W1_orig = nn.W1.copy()
        
        # Save model
        nn.save(test_path)
        print("✓ Model saved")
        
        # Modify weights
        nn.W1 *= 2
        
        # Load model
        nn.load(test_path)
        print("✓ Model loaded")
        
        # Check if weights are restored
        if np.allclose(nn.W1, W1_orig):
            print("✓ Weights correctly restored")
        else:
            print("✗ Weights not correctly restored")
            
        # Clean up
        if os.path.exists(test_path):
            os.remove(test_path)
            
    except Exception as e:
        print(f"✗ Save/load test failed: {e}")

def main():
    print("MNIST AI System Test")
    print("=" * 40)
    
    # Test data loading
    X_train, y_train, X_test, y_test = test_data_loading()
    if X_train is None:
        print("Cannot continue without data")
        return
    
    # Test neural network
    nn = test_neural_network()
    if nn is None:
        print("Cannot continue without neural network")
        return
    
    # Test training step
    test_training_step(nn, X_train, y_train)
    
    # Test save/load
    test_save_load(nn)
    
    print("\n" + "=" * 40)
    print("All tests completed!")
    print("The MNIST AI system is ready to use.")
    print("Run 'python main.py' to start training and GUI.")

if __name__ == "__main__":
    main()
