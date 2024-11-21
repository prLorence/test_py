#!/usr/bin/env python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging verbosity

def test_tensorflow():
    try:
        import tensorflow as tf
        print(f"\nTensorFlow version: {tf.__version__}")
        
        # Test GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print("\nGPU devices found:")
            for gpu in gpus:
                print(f"  {gpu.device_type}: {gpu.name}")
        else:
            print("\nNo GPU devices found. Running on CPU only.")
            
        # Basic operation test
        x = tf.random.normal([2, 3])
        print("\nRandom tensor shape:", x.shape)
        print("Operation test successful!")
        
        return True
        
    except ImportError:
        print("ERROR: TensorFlow is not installed")
        return False
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    test_tensorflow()
