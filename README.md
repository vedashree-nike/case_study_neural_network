## Neural Network

Neural networks are machine learning models that mimic the complex functions of the human brain. These models consist of interconnected nodes or neurons that process data, learn patterns, and enable tasks such as pattern recognition and decision-making.

### Learning in neural networks follows a structured, three-stage process:

Input Computation: Data is fed into the network.
Output Generation: Based on the current parameters, the network generates an output.
Iterative Refinement: The network refines its output by adjusting weights and biases, gradually improving its performance on diverse tasks.

### Layers of Neural Network

Input Layer: This is where the network receives its input data. Each input neuron in the layer corresponds to a feature in the input data.
Hidden Layers: These layers perform most of the computational heavy lifting. A neural network can have one or multiple hidden layers. Each layer consists of units (neurons) that transform the inputs into something that the output layer can use.
Output Layer: The final layer produces the output of the model. The format of these outputs varies depending on the specific task.

![Image](https://github.com/user-attachments/assets/1f79c51d-42fd-41d1-9992-928f48f86489)

## Used Case
With thousands of customers orders generated on a daily basis, tracking delays and recognising patterns become difficult. Being unable to detect common areas and patterns in delay causes three major issues -

1. No right direction in optimizing shipment processes.
2. Potential customers cannot be guaranteed delivery dates in advance.
3. Unsatisfactory customer experience.

## Tech Stack

With the following languages and concepts, the neural network was built.

1. Python
   1.Torch
   2.Pandas
   3.OS
   4.Matplotlib
   5.Numpy
2. Machine Learning fundamentals
3. Neural Networks
4. Linear Algebra
5. Autoencoders

## Approach

Let us consider the following product lifecycle-

Factory -> Port A -> Port B -> DC -> Hubs -> Consumers

There are 9 time periods -

1. Leave Factory - Reach Port A
2. Reach Port A - Leave Port A
3. Leave Port A - Reach Port B
4. Reach Port B - Leave Port B
5. Leave Port B - Reach DC
6. Reach DC - Leave DC
7. Leave DC - Reach Hub
8. Reach Hub - Leave Hub
9. Leave Hub - Reach Consumers
    
If we set a threshold value to all these 9 time periods, if value exceeding such thresholds will be considered as a delay. A delay can be marked as 1 and a no-delay can be marked as 0.
This gives us 9 inputs having values as either 0 or 1. We can feed this into the neural network for further training.

he neural network recieves 9 inputs (0 or 1). This establishes our first input layer of nine neurons.

## Neural Network Model

![Image](https://github.com/user-attachments/assets/22235b2c-948e-4028-b231-3c93c08ea74a)

We have used an Autoencoder model, which is a type of artificial neural network. It works in 2 phases - Encoding and Decoding.
As neural networks work on data with high dimensionality, we use encoders that basically compress the input into a layer with 6 neurons. This gives us the second layer.
The encoder works one more time and compresses the input via three neurons, this is the second hidden layer.
The decoder starts working now, to reconstruct or decompress the data. Giving us another hidden layer of 6 neurons.
Finally, the output layer of 9 neurons is generated.

## Results
The neural network when trained over 30 epochs has the following outputs -

1. The CSV file was generated successfully.
2. The loss during the 1st epoch is 0.25 and this goes down to 0.15 during the 30th epoch.
3. This signifies that the neural network is being trained successfully.
4. Anomalies are detected successfully.
5. Rows with anomalies can be reconstructed based on their deviation and thresholds.
6. The neural network is able to identify which columns (0 indexed) the maximum anomaly/error is being detected.
7. Based on these insights we can plot various graphs.
   
E.g - A graph of MSE against the number of rows can be plotted to identify how much significant deviation occurs in the dataset.

![Image](https://github.com/user-attachments/assets/81a00bd0-0187-4b52-829a-2c1dc930cffd)

## Code Setup
1. Download ```neural_network.py```
2. Download python
3. Install the required packages
4. Run ```python3 neural_network.py```

## Sample Output

--- Starting Anomaly Detection with bin.csv ---
Successfully loaded data of shape: (10000, 9)
Starting training on cpu...
Epoch 1/30, Loss: 0.248623
