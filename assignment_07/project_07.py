# --- World Happiness Agent ---

# Task 1: Define Your Tools
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr
from smolagents import tool

# Shared global DataFrame, filled in by load_happiness_data
df = None

DATA_DIR = Path("happiness_project")

COLUMN_NAMES = {
    "Country": "country",
    "Regional indicator": "region",
    "Happiness score": "happiness_score",
    "Ladder score": "happiness_score",   # the 2024 file uses this name
    "GDP per capita": "gdp_per_capita",
}


@tool
def load_happiness_data() -> dict:
    """Load the World Happiness dataset into memory.

    Loads and merges all of the yearly CSV files (2015-2024) from the
    happiness_project folder into a single table with one row per country per
    year. The table is stored in memory so that the other tools can use it.
    This tool must be called before any of the other tools. The table has the
    columns country, region, happiness_score, gdp_per_capita, and year. The
    other columns, such as Social support, keep their original names.

    Returns:
        A dict with "shape" (the number of rows and columns) and "columns" (the
        list of column names), or an error dict if no data files are found.
    """
    global df

    frames = []
    for path in sorted(DATA_DIR.glob("world_happiness_*.csv")):
        year_df = pd.read_csv(path, sep=";", decimal=",")
        year_df = year_df.rename(columns=COLUMN_NAMES)
        year_df["year"] = int(path.stem.split("_")[-1])   # "world_happiness_2020" -> 2020
        frames.append(year_df)

    if not frames:
        return {"error": f"No world_happiness_*.csv files found in '{DATA_DIR}'."}

    df = pd.concat(frames, ignore_index=True)

    return {"shape": list(df.shape), "columns": df.columns.tolist()}


@tool
def summarize_column(column: str) -> dict:
    """Return descriptive statistics for a single column in the loaded dataset.

    The statistics come from pandas describe(). For a numeric column this
    includes count, mean, std, min, max, and percentiles. For a text column it
    includes count, unique, top, and freq. The dataset must be loaded first with
    load_happiness_data.

    Args:
        column: The name of the column to summarize, for example "happiness_score".

    Returns:
        A dict of descriptive statistics for the column, or an error dict if no
        data is loaded or the column is not found.
    """
    if df is None:
        return {"error": "No data is loaded yet. Call load_happiness_data first."}

    if column not in df.columns:
        return {"error": f"'{column}' is not a column. Options: {df.columns.tolist()}"}

    return df[column].describe().to_dict()


@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Compute the Pearson correlation coefficient and p-value between two numeric columns.

    The Pearson correlation coefficient measures the degree of linear
    correlation between the two columns, from -1 to 1. The dataset must be
    loaded first with load_happiness_data. Rows where either column is missing
    are ignored.

    Args:
        col1: The name of the first numeric column, for example "gdp_per_capita".
        col2: The name of the second numeric column, for example "happiness_score".

    Returns:
        A dict with "col1", "col2", "pearson_r", and "p_value" (rounded to 4
        decimal places), or an error dict if no data is loaded, a column is not
        found, a column is not numeric, or there are too few rows.
    """
    if df is None:
        return {"error": "No data is loaded yet. Call load_happiness_data first."}

    missing = [c for c in [col1, col2] if c not in df.columns]
    if missing:
        return {"error": f"These columns are not in the data: {missing}. Options: {df.columns.tolist()}"}

    not_numeric = [c for c in [col1, col2] if not pd.api.types.is_numeric_dtype(df[c])]
    if not_numeric:
        return {"error": f"These columns are not numeric: {not_numeric}"}

    data = df[[col1, col2]].dropna()
    if len(data) < 2:
        return {"error": "Not enough rows with values in both columns to compute a correlation."}

    r, p = pearsonr(data[col1], data[col2])
    return {
        "col1": col1,
        "col2": col2,
        "pearson_r": round(float(r), 4),
        "p_value": round(float(p), 4),
    }


@tool
def get_top_n_countries(column: str, year: int, n: int = 5) -> list[dict] | dict:
    """Return the top N countries ranked by a given column for a specific year.

    Keeps only the rows for the given year, sorts them by the column from
    highest to lowest, and returns the first n rows. The dataset must be loaded
    first with load_happiness_data.

    Args:
        column: The name of the numeric column to rank by, for example "happiness_score".
        year: The year to look at, for example 2020. The data covers 2015 to 2024.
        n: How many countries to return. Defaults to 5.

    Returns:
        A list of dicts, one per country, each with "country" and the value of
        the requested column, ordered from highest to lowest. Returns an error
        dict if no data is loaded, the column is not found or not numeric, the
        year is not in the data, or n is not a positive number.
    """
    if df is None:
        return {"error": "No data is loaded yet. Call load_happiness_data first."}

    if column not in df.columns:
        return {"error": f"'{column}' is not a column. Options: {df.columns.tolist()}"}

    if not pd.api.types.is_numeric_dtype(df[column]):
        return {"error": f"'{column}' is not numeric, so it cannot be used to rank countries."}

    if year not in df["year"].unique():
        return {"error": f"No data for year {year}. Available years: {sorted(df['year'].unique().tolist())}"}

    if n < 1:
        return {"error": "n must be at least 1."}

    year_df = df[df["year"] == year].dropna(subset=[column])
    top = year_df.sort_values(column, ascending=False).head(n)
    return top[["country", column]].to_dict(orient="records")


# Task 2: Build the Agent
import os

from dotenv import load_dotenv
from smolagents import CodeAgent, OpenAIServerModel

if load_dotenv():
    print("Successfully loaded environment variables from .env")
else:
    print("Warning: could not load environment variables from .env")
api_key = os.getenv("OPENAI_API_KEY")

model = OpenAIServerModel(api_key=api_key, model_id="gpt-4o-mini")

SYSTEM_PROMPT = """
You are a data analyst assistant for the World Happiness dataset.
Use the available tools for loading data, summarizing columns, computing correlations,
and ranking countries. Write Python code directly only when the tools are not sufficient
(for example, when creating custom plots or computing something the tools don't cover).
Be concise and student-friendly in your responses.
"""

agent = CodeAgent(
    tools=[load_happiness_data, summarize_column, compute_correlation, get_top_n_countries],
    model=model,
    instructions=SYSTEM_PROMPT,
    additional_authorized_imports=["pandas", "matplotlib.pyplot", "scipy.stats"],
    max_steps=8,
)


if __name__ == "__main__":
    # Task 3: Run Guided Queries
    # The agent saves its plot to outputs/, so make sure the folder exists
    os.makedirs("outputs", exist_ok=True)

    queries = [
        "Load the happiness data and tell me its shape and column names.",
        "Summarize the happiness_score column.",
        "What is the correlation between gdp_per_capita and happiness_score? Is it statistically significant?",
        "Show me the top 5 happiest countries in 2020.",
        "Plot happiness_score over the years as a line chart, with one line per region. Save the plot to outputs/happiness_by_region.png.",
    ]

    for query in queries:
        print(f"\n--- Query: {query} ---")
        # df is a global that load_happiness_data fills in during the first query.
        # Code the agent writes cannot see it unless we pass it in, so once the data
        # is loaded we hand it over as an additional argument (needed for the plot).
        run_kwargs = {"additional_args": {"df": df}} if df is not None else {}
        response = agent.run(query, reset=False, **run_kwargs)
        print(response)

    # Verify that the plot was saved to disk
    plot_path = os.path.join("outputs", "happiness_by_region.png")
    print(f"\n{plot_path} saved to disk: {os.path.exists(plot_path)}")


    # Task 4: Your Own Questions

    # My query 1
    my_query_1 = "Which 3 countries had the highest gdp_per_capita in 2024?"
    response_1 = agent.run(my_query_1, reset=False)
    print(response_1)
    # Comment: Tool use only. The agent called get_top_n_countries(column="gdp_per_capita",
    # year=2024, n=3) in its first step and got Luxembourg, Ireland, and Singapore.
    # It did not write any analysis code of its own; its second step only passed
    # the tool's result to final_answer.

    # My query 2
    my_query_2 = "Compute the average happiness_score for each year from 2015 to 2024 and tell me which year had the highest average."
    response_2 = agent.run(my_query_2, reset=False, additional_args={"df": df})
    print(response_2)
    # Comment: Code generation. None of the four tools can group the data by year
    # and average it, so the agent wrote its own pandas code in its first step:
    # df.groupby('year')['happiness_score'].mean() with idxmax(), using the df
    # passed in as an additional argument. It called no tools. It found that 2022
    # had the highest average happiness score (about 5.554).


# --- Reflection ---
#
# 1. In Query 3, how did the agent report the correlation? The `compute_correlation` tool
#    returns both a Pearson r and a p-value — did the agent compute and communicate the
#    result sensibly (for example, describing the strength and direction of the
#    relationship, and noting how confident we can be in it)?
#
#    The agent used the compute_correlation tool (no code of its own) and got
#    r = 0.6313 and p = 0.0. Its final answer was a dict:
#    {'pearson_correlation': 0.6313, 'p_value': 0.0, 'significant': True}.
#    It reported the numbers correctly and answered the "is it statistically
#    significant?" part with the significant flag, so the computation was
#    sensible. But it communicated the result only as raw numbers. It did not
#    say in words that the relationship is positive (higher gdp_per_capita goes
#    with higher happiness_score), did not describe how strong it is, and did not
#    explain what the p-value means. The p-value of 0.0 is also the tool's
#    rounding to 4 decimal places, so it means "very small", not exactly zero,
#    and the agent did not point that out.
#
# 2. Did any of the agent's responses surprise you — either by being more capable than
#    you expected, or less? Describe one specific example.
#
#    Yes, Query 5 (the line chart by region) was more capable than I expected. No
#    tool covers this, so the agent wrote its own code in one step: it grouped
#    the df by year and region, took the mean happiness_score, unstacked the
#    result so each region became a line, plotted it, and saved it to
#    outputs/happiness_by_region.png. The file was saved to disk with no errors
#    or retries. It was less capable in Query 1: it treated the result of
#    load_happiness_data() as a DataFrame and got an AttributeError, even though
#    the tool returns a dict. It recovered in the next step by printing the
#    result.
#
# 3. What one additional tool would make this agent meaningfully more useful?
#    Describe what it would do and what kind of question it would help the agent answer.
#    (You do not need to implement it.)
#
#    A get_average_by_group tool. It would take a numeric column (such as
#    happiness_score), a column to group by (such as year or region), and
#    optionally a year, and return the average of that column for each group. It
#    would help the agent answer questions like "Which region has the highest
#    average happiness score?" or "How has the average happiness score changed
#    from year to year?" Right now no tool covers grouping and averaging (my
#    Query 2 in Task 4 asked for exactly this), so the agent has to write its own
#    pandas code each time. A tool would make these common questions faster and
#    more reliable, and it would not depend on the agent being given the
#    DataFrame as an additional argument.