import os

import numpy as np
import pandas as pd
from dash import Dash, html, dash_table, dcc, callback, Output, Input
import plotly.express as px

from bpi_fr_algo_credit_scoring.reader import read_yearly_data
from bpi_fr_algo_credit_scoring.data_pipeline import clean_dataset

# Import of Dataset
default_risk_dataset = read_yearly_data(path=os.getcwd(), default_year=1)
cleaned_default_risk = clean_dataset(
    default_risk_dataset,
    ratio_na_per_features=0.05,
    nb_na_sample_threshold=0,
    ratio_under_oversampled=0.2,
)

# App
app = Dash(__name__)

# App layout
app.layout = html.Div(
    [
        html.H1("Analysis of Default of Companies", style={"text-align": "center"}),
        html.H2("Feature Selection", style={"text-align": "center"}),
        dcc.Dropdown(
            id="dropdown-selection",
            options=cleaned_default_risk.columns.sort_values(),
            value=cleaned_default_risk.columns.sort_values()[0],
            style={"width": "50%", "text-align": "center", "margin": "auto"},
        ),
        html.H2("Data Representation", style={"text-align": "center"}),
        html.Div(
            [
                dash_table.DataTable(
                    id="table-dropdown-output",
                    data={},
                    page_size=20,
                    style_cell={
                        "textAlign": "center",
                        "width": "20%",
                        "margin-left": "10%",
                    },
                ),
                dcc.Graph(
                    figure={},
                    id="boxplot-graph",
                    style={"width": "60%", "margin-left": "10%"},
                ),
            ],
            style={"display": "flex"},
        ),
        html.H2("Correlation Study", style={"text-align": "center"}),
        dcc.Graph(
            figure={},
            id="corr-graph",
        ),
    ]
)


# Add controls to build the interaction
@callback(
    [
        Output(component_id="table-dropdown-output", component_property="data"),
        Output(component_id="table-dropdown-output", component_property="columns"),
    ],
    Input(component_id="dropdown-selection", component_property="value"),
)
def update_table(feat_chosen):
    table = cleaned_default_risk[[feat_chosen]].to_dict("records")
    columns = [{"name": "Values", "id": feat_chosen}]
    return table, columns


@callback(
    Output(component_id="boxplot-graph", component_property="figure"),
    Input(component_id="dropdown-selection", component_property="value"),
)
def update_graph(feat_chosen):
    cleaned_default_risk["Default_str"] = cleaned_default_risk["Default"].map(
        {0: "No-Default", 1: "Default"}
    )
    fig = px.box(data_frame=cleaned_default_risk, y=feat_chosen, color="Default_str")
    fig.update_layout(
        yaxis={"title": "Values"}, xaxis={"title": "Default vs. Non-Default comps"}
    )
    return fig


@callback(
    Output(component_id="corr-graph", component_property="figure"),
    Input(component_id="dropdown-selection", component_property="value"),
)
def update_corr_graph(feat_selected):
    # Selection of a feature and vectorization of its values
    vector_feat_selected = np.array(cleaned_default_risk[feat_selected])
    other_feat_compared = cleaned_default_risk.columns.drop(["Default", "Default_str"])
    # Creation of a vector with correlations results
    results_corr_vector = []
    for feat in other_feat_compared:
        vector_feature = np.array(cleaned_default_risk[feat])
        correlation = np.corrcoef(vector_feat_selected, vector_feature)[0, 1]
        results_corr_vector.append(correlation)
    correlation_df = pd.DataFrame(
        {"Features": other_feat_compared, "Correlations": results_corr_vector}
    )
    correlation_df.sort_values(by="Correlations", ascending=True, inplace=True)
    # Bar Charts with results
    fig = px.bar(data_frame=correlation_df, x="Correlations", y="Features")
    return fig


if __name__ == "__main__":
    app.run(debug=False)
