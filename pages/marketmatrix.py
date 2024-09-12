import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


st.set_page_config(layout="wide")

# ---------------------------------------------------

# prep us_inflation
us_inflation = pd.read_csv("data/us_inflation.csv")

# transpose us_inflation from year + jan, feb, mar, etc columnsto date + month + value columns
us_inflation = us_inflation.melt(
    id_vars=["Year"], var_name="Month", value_name="Inflation"
)

# convert Jan, Feb, etc to 1, 2, etc
us_inflation["Month"] = us_inflation["Month"].apply(
    lambda x: 1
    if x == "Jan"
    else 2
    if x == "Feb"
    else 3
    if x == "Mar"
    else 4
    if x == "Apr"
    else 5
    if x == "May"
    else 6
    if x == "Jun"
    else 7
    if x == "Jul"
    else 8
    if x == "Aug"
    else 9
    if x == "Sep"
    else 10
    if x == "Oct"
    else 11
    if x == "Nov"
    else 12
    if x == "Dec"
    else np.nan
)
# sort by Year and Month + reset index
us_inflation = us_inflation.sort_values(by=["Year", "Month"]).reset_index(drop=True)

st.write(us_inflation)

# ---------------------------------------------------

# prep sp500
sp500_yf = pd.read_csv("data/^GSPC_update.csv")
sp500_bar = pd.read_csv("data/spx_daily_historical-data-09-12-2024.csv")

sp500 = pd.concat([sp500_bar, sp500_yf], ignore_index=True)
sp500["Date"] = pd.to_datetime(sp500["Date"])
sp500 = sp500.drop_duplicates(subset="Date", keep="first")
sp500 = sp500.sort_values("Date").reset_index(drop=True)

# drop columns Change, %Chg and Volume
sp500 = sp500.drop(columns=["Change", "%Chg", "Volume"])

# calculate the monthly aggregated sp500 values for Open, High, Low, Close
sp500["Month"] = sp500["Date"].dt.month
sp500["Year"] = sp500["Date"].dt.year
sp500_monthly = (
    sp500.groupby(["Year", "Month"])
    .agg(
        Open=("Open", "first"),
        High=("High", "max"),
        Low=("Low", "min"),
        Close=("Close", "last"),
    )
    .reset_index()
)

# merge sp500 and us_inflation
sp500_inflation = pd.merge(sp500_monthly, us_inflation, on=["Year", "Month"])

# Replace null or 0 values in the Inflation column with 0.0001
sp500_inflation["Inflation"] = sp500_inflation["Inflation"].replace(
    {0: 0.0001, np.nan: 0.0001}
)

# calculate the monthly growth rate of the S&P 500 in both real (inflation adjusted) and nominal terms
sp500_inflation["Nominal_Growth"] = sp500_inflation["Close"] / sp500_inflation[
    "Close"
].shift(1)
sp500_inflation["Real_Growth"] = (
    sp500_inflation["Close"] / sp500_inflation["Inflation"]
) / sp500_inflation["Close"].shift(1)

st.write(sp500_inflation)

# ---------------------------------------------------

with st.sidebar:
    metric = st.selectbox("Select a metric", options=["Nominal_Growth", "Real_Growth"])

    st.write(metric)

with st.container():
    # lets filter the data to only include up through 2023
    sp500_inflation = sp500_inflation[sp500_inflation["Year"] <= 2023]

    # create a heatmap with a grid of squares
    # each square should have the average return of the starting year and the ending year

    st.write("Data range:")
    st.write(f"Earliest year: {sp500_inflation['Year'].min()}")
    st.write(f"Latest year: {sp500_inflation['Year'].max()}")
    st.write("Sample of data:")
    st.write(sp500_inflation.groupby("Year").first().head())

    # Create a matrix with dimensions from the start to end year of sp500_inflation
    start_year = sp500_inflation["Year"].min()
    end_year = sp500_inflation["Year"].max()
    num_years = end_year - start_year + 1
    matrix = np.full((num_years, num_years), np.nan)

    # Fill the upper triangle of the matrix with the average annualized growth rates
    for i in range(num_years):
        for j in range(i, num_years):
            start_year_val = start_year + i
            end_year_val = start_year + j

            if end_year_val > end_year:
                break  # Stop if we've gone beyond the available data

            start_data = (
                sp500_inflation[(sp500_inflation["Year"] == start_year_val)]
                .sort_values("Month")
                .iloc[0]
            )
            end_data = (
                sp500_inflation[(sp500_inflation["Year"] == end_year_val)]
                .sort_values("Month")
                .iloc[-1]
            )

            if start_data.empty or end_data.empty:
                continue  # Skip this iteration if we don't have data for either year

            start_value = start_data["Close"]
            end_value = end_data["Close"]
            years = j - i + 1  # Add 1 to include both start and end years

            if years == 1:
                # For single year, calculate simple growth rate
                growth_rate = end_value / start_value - 1
            else:
                # For multiple years, calculate compound annual growth rate
                if metric == "Nominal_Growth":
                    growth_rate = (end_value / start_value) ** (1 / (years - 1)) - 1
                else:  # Real_Growth
                    inflation_factor = end_data["Inflation"] / start_data["Inflation"]
                    growth_rate = ((end_value / start_value) / inflation_factor) ** (
                        1 / (years - 1)
                    ) - 1

            matrix[i, j] = growth_rate

    # Remove any rows and columns that are all NaN
    matrix = matrix[~np.isnan(matrix).all(axis=1)][:, ~np.isnan(matrix).all(axis=0)]

    # Update the years list to match the new matrix dimensions
    years = list(range(start_year, start_year + matrix.shape[1]))

    # Create a custom color scale
    colors = [
        (0, "red"),
        (0.35, "pink"),
        (0.5, "white"),
        (0.65, "palegreen"),
        (1, "green"),
    ]

    # Calculate the midpoint for the color scale
    vmin = np.nanmin(matrix)
    vmax = np.nanmax(matrix)
    abs_max = max(abs(vmin), abs(vmax))
    midpoint = 1 / (1 + (abs_max / abs(vmin)))

    # Create a Plotly heatmap
    fig = px.imshow(
        matrix,
        x=years,
        y=years,
        color_continuous_scale=colors,
        color_continuous_midpoint=0,  # Set the midpoint of the color scale to 0
        zmin=-abs_max,  # Set the minimum value for the color scale
        zmax=abs_max,  # Set the maximum value for the color scale
        labels=dict(x="Ending Year", y="Starting Year", color=f"Annualized {metric}"),
        title=f"Annualized {metric} Market Matrix",
    )

    # Customize the layout
    fig.update_layout(
        xaxis=dict(tickmode="linear", tick0=start_year, dtick=5),
        yaxis=dict(tickmode="linear", tick0=start_year, dtick=5),
        coloraxis_colorbar=dict(
            title=f"Annualized {metric}",
            tickformat=".1%",  # Format tick labels as percentages
        ),
        width=1000,  # Increase width
        height=800,  # Increase height
        autosize=False,  # Disable autosize to use our custom dimensions
    )

    # Update hover template to show more information
    fig.update_traces(
        hovertemplate="Starting Year: %{y}<br>Ending Year: %{x}<br>Annualized "
        + metric
        + ": %{z:.2%}<extra></extra>"
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # full data
    st.write(sp500_inflation)
