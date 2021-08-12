   
## 1. Add the necessary CUDA binaries to the $PATH environment variable:

    add the following lines to the ~/.bashrc file in your home directory:

    export PATH=/usr/local/cuda-9.2/bin:$PATH

    export LD_LIBRARY_PATH=/usr/local/cuda-9.2/lib64:$LD_LIBRARY_PATH
    
    save, and in a terminal run:
    
    source ~/.bashrc

## 2. Create a python virtual environment to install the necessary python packages.
In a terminal, from a folder of your choice, run:

    ```virtualenv --no-site-packages -p python3 venv```
    
    ```source venv/bin/activate```

## 3. Install the python packages (please note: this operation requires at least 1.3GB of disk space / disk quota!)

    in a terminal run:
    
    ```pip install --upgrade pip```
    
    ```pip install numpy Pillow```
    
    ```pip install torch==1.4.0+cu92 torchvision==0.5.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html```
    

## 4. Try importing torch and torchvision from within the virtual environment. From a python interpreter:

    ```import torch, torchvision```
