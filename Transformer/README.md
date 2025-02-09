# Transformer

## Setup Interpreter Conda in Visual Studio Code
1. **Install Anaconda**
   - https://www.anaconda.com/download

2. **Open Command Palette**
   - Klik `Ctrl + Shift + P` (atau `Cmd + Shift + P` di Mac) for open Command Palette.

3. **Choose Interpreter**
   - Klik `Python: Select Interpreter` in Command Palette, choose option.

4. **Choose Environment Conda**
   - List interpreter, choose environment Conda, example `conda-env:<environment_name>`.

5. **If Environment Not Visible**
   - Choose `Enter interpreter path...`, setup location Python environment Conda:
     - **Windows**: `C:\Users\<username>\Anaconda3\envs\<environment_name>\python.exe`
     - **Mac/Linux**: `/Users/<username>/anaconda3/envs/<environment_name>/bin/python`

6. **Verification**
   - Verification environment in Visual Studio Code.

# Transformer Model

This project is a text classification model built using Python, FastAPI, and scikit-learn. The model is trained to classify text data into predefined intents using Bert, and it serves predictions via a FastAPI backend.

## Table of Contents
1. [Installation](#installation)
2. [Setup](#setup)
3. [Running the Server](#running-the-server)
4. [Testing the API](#testing-the-api)

## Installation

To get started with this project, you need to install the required dependencies.

1. Clone the repository:

    ```bash
    git clone <repository-url>
    cd <project-directory>
    ```

2. Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Install FastAPI, Uvicorn, and Joblib:

    ```bash
    pip install fastapi uvicorn joblib
    ```

## Setup

Before running the server, make sure you have the necessary data and scripts:

- **data3.csv**: This file contains the data that will be used to train the text classification model. It should have at least two columns:
  - `text`: The text data.
  - `intent`: The corresponding intent for each piece of text.

- **train.py**: This script is used to train the text classification model and save it to a `.pth` file. The model uses `BERT` for text vectorization and `Transformer` for classification.

- **app.py**: This file contains the FastAPI application that exposes an API endpoint for making predictions using the trained model.

### Running the Server

After setting up the environment, you can run the FastAPI server locally using Uvicorn.

1. **Start the server**:

    Run the following command to start the FastAPI app on `http://127.0.0.1:8000`:

    ```bash
    uvicorn app:app --reload
    ```

    Alternatively, if you're using a different entry point for the FastAPI app:

    ```bash
    uvicorn main:app --reload
    ```

    This will reload the server automatically upon code changes.

## Testing the API

Once the FastAPI server is running, you can test the API by sending HTTP requests to it. You can use [Hoppscotch](https://hoppscotch.io/) or [Postman](https://www.postman.com/) to make requests.

### Using Hoppscotch to Test the API

1. Go to [Hoppscotch.io](https://hoppscotch.io/) in your web browser.

2. **Select the method**: Choose `POST` from the dropdown menu next to the URL input.

3. **Enter the API URL**: In the URL bar, type:

    ```
    http://127.0.0.1:8000/predict
    ```

4. **Set the request body**: In the body section, select the `JSON` option and enter the following:

    ```json
    {
      "text": "Halo, selamat datang!"
    }
    ```

5. **Send the request**: Click on the `Send` button.

6. **Check the response**: You will receive a response from the server, such as:

    ```json
    {
      "intent": "greeting"
    }
    ```

This will show the predicted intent for the input text.