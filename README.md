# 2D coordinate mapping with PyTorch

This is a small example project to visualize different mapping functions of 2D coordinates learned by a neural network.

## 1. Setup
Install all requirements with `pip install -r requirements.txt`

## 2. Collect training data
You can collect training data with the `record_data.py`. Options for learnable
functions are 'increase', 'decrease' and 'reverse' and can be selected with the
argument `-f`. </br>
Example usage:
```bash
# This will create the file 'data.csv' in the current directory
python record_data.py -f 'reverse'
```
After the pygame window has opend, move your cursor inside the window to
collect training samples.
This process runs until the window is closed.

## 3. Train the model
Die gesammelten Trainingsdaten können nun genutzt werden um ein neuronales Netz
zu trainieren.

The command
```bash
python train.py
```
expects the CSV file with the training data in the current directory. After training, the network is saved under the path './saved_model.pt'. <br>
The path to the CSV file and the folder to save the neural network can be specified with the arguments `--data` and `--output_dir``.

## 4. Visualize the predicitons
To visualize the predictions of the network, run the following command:
```bash
python visualize.py -f 'reverse'
```
This will visualize the predicted coordinated of the learned mapping function
with a red dot and the actual position with a green dot.
This command expects the `saved_model.pt` in the current directory. If you want
to change the path to the trained model or the function used to visualize the 
ground truth coordinates use the arguments `--model_path` or `-f`.