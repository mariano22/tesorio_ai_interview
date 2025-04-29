"""
Data Science Health Check Script.

This module performs a health check for data science operations including:
- Creating sample DataFrames
- Reading CSV files
- Converting data to Markdown format
- Analyzing data with OpenAI API
"""
import click
import numpy as np
import pandas as pd
from openai import OpenAI

# Assuming 'app' directory is in the same root or accessible via PYTHONPATH
try:
    from app import prompts
except ImportError:
    print("Error: Could not import prompts from 'app' directory.")
    print("Ensure 'app/prompts.py' exists and the 'app' directory is accessible.")
    prompts = None # Set to None to handle gracefully later

# Attempt to load the API key from environment variables
# Ensure the OPENAI_API_KEY environment variable is set before running.
try:
    CLIENT = OpenAI()
except (ValueError, TypeError) as api_error:
    print(f"Error initializing OpenAI client: {api_error}")
    print("Please ensure the OPENAI_API_KEY environment variable is set.")
    CLIENT = None

# Define the path to the sample CSV file
# Assumes the script is run from the project root directory
# where 'data' and 'genai-tech-screen' are siblings.
CSV_FILE_PATH = 'data/sample_data.csv'


@click.command()
def health_check():
    """
    Performs a data science health check:
    1. Generates a sample DataFrame.
    2. Reads a sample CSV file from the 'data' directory.
    3. Converts both DataFrames to Markdown.
    4. Sends them to OpenAI API for analysis using prompts from app.prompts.
    5. Prints the response.
    """
    if not CLIENT:
        click.echo(
            "OpenAI client not initialized. Cannot proceed.", err=True
        )
        return

    if not prompts:
        click.echo(
            "Prompts module not loaded. Cannot proceed.", err=True
        )
        return

    click.echo("Performing data science health check...")

    # --- 1. Generate sample data and DataFrame ---
    click.echo("Generating sample DataFrame...")
    data = np.random.rand(5, 3)  # Create a 5x3 array of random floats
    columns = ['Feature_A', 'Feature_B', 'Target']
    generated_df = pd.DataFrame(data, columns=columns)
    click.echo("Sample DataFrame created:")
    click.echo(generated_df.to_string())
    click.echo("-" * 30)

    # --- 2. Read sample CSV file ---
    csv_df = None
    csv_markdown_table = None
    click.echo(f"Attempting to read CSV file: {CSV_FILE_PATH}...")
    try:
        csv_df = pd.read_csv(CSV_FILE_PATH)
        click.echo("CSV file read successfully:")
        click.echo(csv_df.to_string())
        click.echo("-" * 30)
        # Convert CSV DataFrame to Markdown
        click.echo("Converting CSV DataFrame to Markdown...")
        csv_markdown_table = csv_df.to_markdown(index=False)
        click.echo("CSV Markdown Table:")
        click.echo(csv_markdown_table)
    except FileNotFoundError:
        click.echo(f"Warning: CSV file not found at {CSV_FILE_PATH}.", err=True)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as csv_error:
        click.echo(f"Error reading CSV file: {csv_error}", err=True)
    click.echo("-" * 30)


    # --- 3. Convert generated DataFrame to Markdown ---
    click.echo("Converting generated DataFrame to Markdown...")
    generated_markdown_table = generated_df.to_markdown(index=False)
    click.echo("Generated Markdown Table:")
    click.echo(generated_markdown_table)
    click.echo("-" * 30)

    # --- 4. Send to OpenAI API ---
    click.echo("Sending data to OpenAI API for analysis...")

    # Construct the prompt using the function from app.prompts
    prompt = prompts.get_user_prompt(
        generated_markdown=generated_markdown_table,
        csv_markdown=csv_markdown_table
    )

    try:
        response = CLIENT.chat.completions.create(
            model="gpt-3.5-turbo",  # Or your preferred model
            messages=[
                {"role": "system", "content": prompts.SYSTEM_MESSAGE},
                {"role": "user", "content": prompt}
            ]
        )
        # --- 5. Print the response ---
        ai_response = response.choices[0].message.content
        click.echo("OpenAI API Response:")
        click.echo(ai_response)

    except (ValueError, TypeError, KeyError) as openai_error:
        click.echo(f"Error calling OpenAI API: {openai_error}", err=True)

    click.echo("-" * 30)
    click.echo("Health check complete.")


if __name__ == '__main__':
    health_check()
