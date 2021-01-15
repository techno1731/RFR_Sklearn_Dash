# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import math
import main.model as mod

import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State

from settings import config, about

# ----------------------------------------------------------------------------
# Data Preparation for the App
# ----------------------------------------------------------------------------


# Load the model

model = mod.load_model()

# defining the column names
cols = [
    "MPG",
    "Cylinders",
    "Displacement",
    "Horsepower",
    "Weight",
    "Acceleration",
    "Model_Year",
    "Origin",
]
# reading the .data file using pandas, sep = " "
df = pd.read_csv(config.root+"/data/auto-mpg.data",
    names=cols,
    na_values="?",
    comment="\t",
    sep=" ",
    skipinitialspace=True,
)

data = df.drop("MPG", axis=1)

data_prep = mod.full_preprocess(data)

df_feature_importances = pd.read_csv(config.root+"/data/features_ranked.csv")
df_feature_importances = df_feature_importances.sort_values(
    "importance", ascending=False
)

# We create a Features Importance Bar Chart
fig_features_importance = go.Figure()

# making a copy of the dataframe
fig_features_importance.add_trace(
    go.Bar(
        x=df_feature_importances["feature"],
        y=df_feature_importances["importance"],
        marker_color="rgb(171, 226, 251)",
    )
)
fig_features_importance.update_layout(
    title_text="<b>Features Importance of the model<b>", title_x=0.5
)
# The command below can be activated in a standard notebook to display the chart
# fig_features_importance.show()

# record the name, min, mean and max of the three most important features
slider_1_label = df_feature_importances["feature"][0]
slider_1_min = math.floor(data[slider_1_label].min())
slider_1_mean = round(data[slider_1_label].mean())
slider_1_max = round(data[slider_1_label].max())

slider_2_label = df_feature_importances["feature"][1]
slider_2_min = math.floor(data[slider_2_label].min())
slider_2_mean = round(data[slider_2_label].mean())
slider_2_max = round(data[slider_2_label].max())

slider_3_label = df_feature_importances["feature"][2]
slider_3_min = math.floor(data[slider_3_label].min())
slider_3_mean = round(data[slider_3_label].mean())
slider_3_max = round(data[slider_3_label].max())

# ----------------------------------------------------------------------------
# Dash App Styling Begin
# ----------------------------------------------------------------------------

# App Instance
app = dash.Dash(name=config.name, 
                assets_folder=config.root+"/app/static", 
                external_stylesheets=[dbc.themes.SLATE, 
                                      config.fontawesome])
app.title = config.name

# Navbar
navbar = dbc.Nav(className="nav nav-pills", children=[
    ## logo/home
    dbc.NavItem(html.Img(src=app.get_asset_url("logo.png"), height="40px")),
    ## about
    dbc.NavItem(html.Div([
        dbc.NavLink("How-to", href="/", id="about-popover", active=False),
        dbc.Popover(id="about", is_open=False, target="about-popover", children=[
            dbc.PopoverHeader("How it works"), dbc.PopoverBody(about.txt)
        ])
    ])),
    ## links
    dbc.DropdownMenu(label="Links", nav=True, children=[
        dbc.DropdownMenuItem([html.I(className="fa fa-linkedin"), "  linkedin"], href=config.contacts, target="_blank"), 
        dbc.DropdownMenuItem([html.I(className="fa fa-github"), "  Code"], href=config.code, target="_blank"),
        dbc.DropdownMenuItem([html.I(className="fa fa-folder-open"), "  Portfolio"], href=config.portfolio, target="_blank")
    ])
])

# ----------------------------------------------------------------------------
# Styling variables and definitions  
# ----------------------------------------------------------------------------

align_center = {'textAlign' : 'center'}

prediction_card = [
    dbc.CardHeader(html.H3("Prediction Result",style=align_center)),
    dbc.CardBody(
        [
            html.H3(id="prediction_result", className="card-text", style=align_center),
        ]
    ),
]
graph_card = [
    dbc.CardHeader(html.H3("Features Rank", style= align_center)),
    dbc.CardBody(
        [
            html.Div(children=[
           
            # We display the most important feature's name
            dcc.Graph(figure=fig_features_importance)])
        ]
    ),
]
sliders_card = [
    dbc.CardHeader(html.H3("Top Features Control", style= align_center)),
    dbc.CardBody(
        [
            html.Div(children=[
                    # The Dash Slider is built according to Feature #1 ranges
                    html.H4(slider_1_label, style= align_center),
                    dcc.Slider(
                        id="X1_slider",
                        min=slider_1_min,
                        max=slider_1_max,
                        step=100,
                        value=slider_1_mean,
                        marks={
                            i: "{} bars".format(i) for i in range(slider_1_min, slider_1_max + 1, 500)
                        },
                    ),
                    # The same logic is applied to the following names / sliders
                    html.H4(slider_2_label, style= align_center),
                    dcc.Slider(
                        id="X2_slider",
                        min=slider_2_min,
                        max=slider_2_max,
                        step=1,
                        value=slider_2_mean,
                        marks={
                            i: "{}°".format(i) for i in range(slider_2_min, slider_2_max + 1)
                        },
                    ),
                    html.H4(slider_3_label, style= align_center),
                    dcc.Slider(
                        id="X3_slider",
                        min=slider_3_min,
                        max=slider_3_max,
                        step=10,
                        value=slider_3_mean,
                        marks={
                            i: "{}".format(i) for i in range(slider_3_min, slider_3_max + 1, 10)
                        },
                    ),
                ]),
        ]
    ),
]
# ----------------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------------

app.layout = dbc.Container(fluid=True, children=[
    ## Top
    html.H1(config.name, id="nav-pills", style= align_center),
    navbar,
    html.Br(),

    ## Body
    dbc.Row([
        # Grap Column
        dbc.Col(
            dbc.Card(graph_card, color="primary", inverse=True)
        ),
        dbc.Col(children=[
            dbc.Row([
                dbc.Col(
                    dbc.Card(sliders_card, color="primary", inverse=True)
                )
            ]),
            dbc.Row([
                dbc.Col(
                    dbc.Card(prediction_card, color="primary", inverse=True)
                )
            ])
        ])
    ])
])
"""
        # Sliders + Results Column
        dbc.Col([
            # Slider Row
            dbc.Row([
                # Slider Column
                dbc.Col(
                    "Slider",
                #    dbc.Card(prediction_card, color="primary", inverse=True)
                ),
            ]),
            dbc.Row(
                # Prediction Column
                dbc.Col(
                    "Prediction",
                   # dbc.Card(prediction_card,color="primary", inverse=True)
                )
            )
        ])

    ])
])
"""
# ----------------------------------------------------------------------------
# Dash @Callbacks 
# ----------------------------------------------------------------------------

# Python functions for about navitem-popover
@app.callback(output=Output("about","is_open"), inputs=[Input("about-popover","n_clicks")], state=[State("about","is_open")])
def about_popover(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(output=Output("about-popover","active"), inputs=[Input("about-popover","n_clicks")], state=[State("about-popover","active")])
def about_active(n, active):
    if n:
        return not active
    return active


# The callback function will provide one "Ouput" in the form of a string (=children)
@app.callback(
    Output(component_id="prediction_result", component_property="children"),
    # The values correspnding to the three sliders are obtained by calling their id and value property
    [
        Input("X1_slider", "value"),
        Input("X2_slider", "value"),
        Input("X3_slider", "value"),
    ],
)

# The input variable are set in the same order as the callback Inputs


def update_prediction(X1, X2, X3):

    # We create a NumPy array in the form of the original features
    # ["Pressure","Viscosity","Particles_size", "Temperature","Inlet_flow", "Rotating_Speed","pH","Color_density"]
    # Except for the X1, X2 and X3, all other non-influencing parameters are set to their mean
    input_X = {
        "Cylinder": [4, 8, 6],
        "Displacement": [data["Displacement"].mean(), 160, 165.5],
        "Horsepower": [X3, 130, 98],
        "Weight": [X1, 3150, 2600],
        "Acceleration": [data["Acceleration"].mean(), 18, 16],
        "Model Year": [X2, 80, 78],
        "Origin": [2, 3, 1],
    }

    # Prediction is calculated based on the preprocessed input_X array
    prediction = mod.predict_mpg(input_X, model)

    # And retuned to the Output of the callback function
    return "MPG: {}".format(round(list(prediction)[0],ndigits=3))


if __name__ == "__main__":
    app.run_server()
