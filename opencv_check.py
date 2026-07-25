import cv2
import numpy as np

# Function to check if OpenCV is compiled with CUDA support
def is_opencv_with_cuda_support():
    try:
        # Print version information (this might indicate CUDA support)
        print("OpenCV Version:", cv2.__version__)
        
        # Check if CUDA-capable devices are available
        num_devices = cv2.cuda.getCudaEnabledDeviceCount()
        if num_devices == 0:
            print("\nNo CUDA-capable device found.")
            return False
        
        print(f"\n{num_devices} CUDA-capable device(s) found.")
        
        # Set the CUDA device to use (e.g., device ID 0)
        cv2.cuda.setDevice(0)

        # Create a small image and upload it to the GPU
        img_size = (64, 64)
        gpu_mat = cv2.cuda_GpuMat()
        gpu_mat.upload(np.zeros(img_size, dtype=np.uint8))

        # Perform a basic operation on the GPU (e.g., converting to grayscale)
        gray_gpu = cv2.cuda.cvtColor(gpu_mat, cv2.COLOR_GRAY2GRAY)

        # Download the result back to CPU
        gray_cpu = gray_gpu.download()
        
        # Print the dimensions of the processed image (sanity check)
        print(f"Processed Image Size: {gray_cpu.shape}")

        print("\nOpenCV is successfully using CUDA.")
        return True

    except Exception as e:
        print(f"\nError during CUDA check:\n{e}")
        return False

# Main function to execute the check
def main():
    print("Checking OpenCV CUDA Support...")
    result = is_opencv_with_cuda_support()
    
    if result:
        print("\nOpenCV appears to be compiled with and using CUDA support.")
    else:
        print("\nOpenCV might not be compiled with or is not using CUDA support.")

# Execute the main function
if __name__ == "__main__":
    main()
