# Minimal Dash dashboard for Cryptobot
try:
    from dash import Dash, html, dcc
    from dash.dependencies import Input, Output
    import dash_bootstrap_components as dbc
except ImportError as e:
    raise ImportError(
        "Dash is required for the dashboard. Install with:\n"
        "    pip install dash dash-bootstrap-components\n"
    ) from e

def build_app() -> "Dash":
    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.DARKLY],
        title="Cryptobot Dashboard",
    )

    app.layout = dbc.Container(
        [
            html.H2("Cryptobot Dashboard"),
            html.P("Live analytics coming soon…"),
            dcc.Interval(id="tick", interval=2000, n_intervals=0),
            html.Div(id="status", children="Ready"),
        ],
        fluid=True,
    )

    @app.callback(Output("status", "children"), Input("tick", "n_intervals"))
    def _heartbeat(n):
        return f"Heartbeat: {n}"

    return app

def run_dashboard(host: str = "0.0.0.0", port: int = 8050, debug: bool = False) -> None:
    app = build_app()
    app.run_server(host=host, port=port, debug=debug)

if __name__ == "__main__":
    run_dashboard()

