# Unmanned Aerial Vehicle and Unmanned Charging Car Cooperative Control System

## Project Overview

This project implements a reinforcement learning control algorithm model to facilitate collaboration between Unmanned Aerial Vehicles (UAVs) and unmanned charging vehicles. The system utilizes reinforcement learning models and simulation environments to assist users in managing UAVs and unmanned charging vehicles to complete observation tasks at designated locations. UAVs are extensively employed for observation tasks in forests, oceans, and national parks. The introduction of unmanned charging vehicles offers essential charging support for UAVs, significantly enhancing their endurance and operational duration, thereby improving work efficiency and broadening application scenarios.

## System Features

This system offers the following key functionalities:

- **Simulation Scene Generation**: Create a virtual environment that includes UAVs, unmanned charging cars, charging points, and observation points.
- **Model Training**: Train the collaborative control strategy for UAVs and unmanned charging cars using reinforcement learning algorithms.
- **Model Application**: Apply the trained model to real-world scenarios, enabling the UAVs and unmanned charging cars to perform collaborative tasks.
- **Control Strategy Generation**: Generate movement strategies, observation strategies, and charging strategies for both UAVs and unmanned charging cars.
- **Visualization Interface**: Provide a visualization interface to help users monitor and control the system's operations.

## System Requirements

- **Operating System**: Windows 7 / Windows 10 / Windows 11
- **Minimum Hardware Configuration**:
  - CPU: 2GHz or higher
  - RAM: 4GB or higher
- **Development Technologies**:
  - Programming Language: Python
  - Frameworks: Pytorch (for semantic segmentation models), PyQt5 (for the visualization interface)

## Setup Instructions

1. **Clone the Project**:
    Clone the project repository to your local machine:

   ```bash
   git clone https://github.com/jizheDou/HaDMC.git
   ```

2. **Install Dependencies**:
    Install the required Python libraries using `pip`:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Simulation**:
    Run the `HaDMC/train.py` script to start the model training. This will begin training the reinforcement learning model and generate the control strategies.

   ```bash
   python HaDMC/train.py
   ```

## File Structure

![image-20250420011406632](C:\Users\Yvonne\AppData\Roaming\Typora\typora-user-images\image-20250420011406632.png)

```plaintext
|-- HaDMC
    |-- event
    |-- map
    |-- scenario
    |-- world
    |-- vehicle
    |-- model
    |-- main.py
    |-- ui_HaDMC.py
    |-- UIHaDMC.py
```

## Model Architecture

![image-20250413173845186](C:\Users\Yvonne\AppData\Roaming\Typora\typora-user-images\image-20250413173845186.png)

This system adopts a reinforcement learning-based cooperative control algorithm for UAVs and unmanned charging vehicles. The model consists of two main components: an **Encoder** and a **Decoder**:

- **Encoder**: Learns the relationship between the environment and latent actions using reinforcement learning algorithms.
- **Decoder**: Decodes the latent actions using an Adversarial AutoEncoder (AAE) and a mapping table to generate strategies for both the UAV and the unmanned charging vehicle.

## Simulation Environment

The simulation environment is designed based on real-world cooperative observation scenarios involving UAVs and unmanned charging vehicles. The environment includes:

- UAVs
- Unmanned Charging Vehicles
- Charging Stations
- Observation Points

The system is implemented in Python to enable coordination between UAVs, unmanned charging vehicles, and other environmental elements.

## System Requirements

- **Minimum Hardware Configuration**:
  - CPU: 2GHz or higher
  - RAM: 4GB or more

## FAQ

### 1. How can I modify the simulation environment?

You can modify the simulation environment by editing the files located in the `map`, `scenario`, and `world` directories.

### 2. What should I do if I encounter an out-of-memory error during training?

Try reducing the size of the training data or consider upgrading your system's hardware configuration.

### 3. How can I view training logs?

Training logs are printed in the terminal during execution. Additionally, you can check the `logs` folder for saved log files.

## Contact

For any issues or questions, please contact the project maintainer: **jizheDou**