# app/prompts.py

SYSTEM_MESSAGE = "You are a helpful data analyst assistant."

def get_user_prompt(generated_markdown, csv_markdown=None):
    """
    Generates the user prompt for the OpenAI API call.

    Args:
        generated_markdown (str): Markdown table for the generated data.
        csv_markdown (str, optional): Markdown table for the CSV data.
                                      Defaults to None.

    Returns:
        str: The formatted user prompt.
    """
    prompt_parts = [
        "Here are two sample datasets in Markdown format.",
        "\nDataset 1 (Generated):",
        generated_markdown
    ]

    if csv_markdown:
        prompt_parts.extend([
            "\nDataset 2 (From CSV):",
            csv_markdown,
            "\nPlease provide a brief description of each dataset, "
            "mentioning the features and potential target variables. "
            "What kind of simple analysis could be performed on each, "
            "or potentially by combining them?"
        ])
    else:
        prompt_parts.append(
            "\nPlease provide a brief description of this dataset, "
            "mentioning the features and potential target variable. "
            "What kind of simple analysis could be performed on this data?"
        )

    return "\n\n".join(prompt_parts)

