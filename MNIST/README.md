# MNIST Classification with PyTorch

This project demonstrates a simple Convolutional Neural Network (CNN) for classifying the MNIST dataset using PyTorch. The model is trained and evaluated using a CI/CD pipeline set up with GitHub Actions.

## Project Structure

├── .github
│ └── workflows
│ └── ci-cd.yml
├── data
│ └── (raw data will be stored here)
├── models
│ └── (saved models will be stored here)
├── train_and_test.py
├── requirements.txt
└── README.md


## Requirements

- Python 3.8 or higher
- PyTorch
- Torchvision
- Numpy

## Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/sdeepthikumar/AI.git
   cd <repository-directory>
   ```

2. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the dependencies:**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## Running the Model

To train and evaluate the model, run the following command:

```bash
python train_and_test.py
```

This script will:

- Download and preprocess the MNIST dataset.
- Train a simple CNN model for one epoch.
- Evaluate the model on the test dataset.
- Save the trained model in the `models` directory with a timestamp.

## CI/CD Pipeline

The project includes a GitHub Actions workflow (`.github/workflows/ci-cd.yml`) that automates the following tasks on every push or pull request:

- Set up the Python environment.
- Install dependencies.
- Run the training and testing script to ensure the model meets specified criteria.

## Model Validation

The model is validated to ensure:

- It has fewer than 100,000 parameters.
- It accepts 28x28 input images.
- It outputs 10 classes.
- It achieves an accuracy of more than 80% on the test dataset.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
