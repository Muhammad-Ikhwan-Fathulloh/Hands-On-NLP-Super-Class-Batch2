# Introduction to LLM (Large Language Model)

- [LLM](Introduction_LLM.md): Penjelasan mendalam mengenai **Introduction to LLM (Large Language Model)**.

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

# LLM Model

This project is a LLM model built using Python, FastAPI, and Pytorch. The model is trained to predict name data and it serves predictions via a FastAPI backend.

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

3. Install FastAPI, Uvicorn:

    ```bash
    pip install fastapi uvicorn
    ```

## Setup

Before running the server, make sure you have the necessary data and scripts:

- **llm**: This file contains the data that will be used to train the llm model. It should have at least two columns:
  - `instruction`: The instruction.
  - `input_data`: The input_data of user.
  - `response`: The response of promt.

- **model**: This script is used to train the llm model and save it to a `rubythalib33/llama3_1_8b_finetuned_bahasa_indonesia` file.

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
    http://127.0.0.1:8000/chat
    ```

4. **Set the request body**: In the body section, select the `JSON` option and enter the following:

    ```json
    {
      "instruction": "Tulis artikel singkat tentang manfaat kecerdasan buatan (AI).",
      "input_data": "Tuliskan dengan gaya bahasa yang mudah dipahami oleh orang awam."
    }
    ```

5. **Send the request**: Click on the `Send` button.

6. **Check the response**: You will receive a response from the server, such as:

    ```json
    {
      "response": "Kecerdasan buatan (AI) adalah teknologi yang memungkinkan komputer dan mesin melakukan tugas-tugas yang biasanya memerlukan kecerdasan manusia, seperti pemecahan masalah, pengambilan keputusan, dan pembelajaran.  AI memiliki banyak manfaat, termasuk meningkatkan efisiensi, mengurangi biaya, dan meningkatkan keakuratan.",
    }
    ```

This will show the predicted intent for the input text.