# DealScout: AI Commercial Real Estate Assistant

This autonomous agent, named DealScout, is designed to assist commercial real estate investors in identifying, evaluating, and managing properties. It leverages advanced AI tools and external APIs to provide real-time market data, calculate crucial financial metrics like Cap Rates, maintain an active property watchlist, and ensure secure operations by requiring human approval for any destructive actions.

## Setup Instructions

To get started with DealScout, follow these steps:

1. **Install Dependencies:**
    Ensure you have Python 3.9+ installed. Then, install the required libraries using pip:

    ```bash
    pip install -r requirements.txt
    ```

2.  **Set Up API Keys:**
    DealScout requires API keys for its functionality. Obtain your keys for Gemini and Serper, and then set them as environment variables. You can do this by exporting them in your terminal session or by adding them to a `.env` file.

    **For Terminal Session (replace with your actual keys):**

    ```bash
    set GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
    set SERPER_API_KEY="YOUR_SERPER_API_KEY"
    ```
    **`.env` File (Recommended for Local Development):**
    Use the `python-dotenv` library to load environment variables from a `.env` file. This file should **never** be committed to version control.

     Install `python-dotenv`:
    ```bash
    pip install python-dotenv
    ```
    Create a file named `.env` in the root directory of your project with the following content (replace `YOUR_GEMINI_KEY` and `YOUR_SERPER_KEY` with your actual keys):
    ```
    GEMINI_API_KEY="YOUR_GEMINI_KEY"
    SERPER_API_KEY="YOUR_SERPER_KEY"
    ```

    **For Colab Notebooks:**
    In Google Colab, you can add your API keys to the 'Secrets' manager (the '🔑' icon in the left panel) and name them `GEMINI_API_KEY` and `SERPER_API_KEY`.

## Execution Guide

Depending on how the DealScout agent is structured, you can run it using one of the following methods:

1.  **Run the Main Python Application:**
    If the agent's core logic is in `main.py`:

    ```bash
    python main.py
    ```

2.  **Run a Streamlit Web Application:**
    If the agent includes a Streamlit interface (e.g., in `app.py`):
    ```bash
    python -m streamlit run app.py
    ```
    This will typically open a new tab in your web browser with the DealScout application.
