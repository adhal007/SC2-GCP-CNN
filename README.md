# SC2-GCP-CNN

## Local Implementation of CNN neural network for StarCraft II using Keras
2 files to run the CNN neural network for SCII game data on command prompt.
1. ```model.py``` : This file defines the model architecture for the neural network. It currently uses 3 hidden layers of CNN with 32, 64, 128 architecture followed by Dense MLP with softmax implementation. It is imported into the second file
2. ``` task.py ```: This file builds the training data with the following arguments:<br/>
    a. ```--model-name```: ```(str)``` type  - Give a model name for saving<br/>  
    b. ```--model-dir``` : ```(str)```type   - Give model directory where raw input data is located   <br/>    
    c. ```--batch-size```: ```(int)```type   - Specify the batch size for each epoch to the run the training and evaluate the results<br/>  
    d. ```--test-split```: ```(float)```type - a value between (0, 1) for training testing data splitting <br/>  
    e. ```--seed```: ```(int)``` type        - set seed for the run <br/>  
    f. ```--increment```: ```(int)``` type   - additional argument for iterating through n files at a time when training and testing.
 
## Running on command prompt:
1. Ensure Python is in the path of command prompt for windows, For linux users configure the .bashrc script to add python to path, similar for mac users <br/>
2. Go to directory where the ```task.py``` file is located. </br>
3. Run ``` python task.py --model-name "myCNNModel" --model-dir "C:/Users/..." --batch-size 25 --test-split 0.3 --seed 42 --increment 25 ```
