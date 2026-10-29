from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_dashboard_navigation_and_applicant_prediction():
    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
    assert not app.exception
    assert app.title[0].value == "Credit Risk Prediction & Explainability"

    for page in [
        "Applicant Information",
        "Model Performance",
        "Explainability",
        "Project Information",
        "Overview",
    ]:
        app.radio[0].set_value(page).run(timeout=30)
        assert not app.exception

        if page == "Applicant Information":
            app.button[0].click().run(timeout=30)
            assert not app.exception
            assert [metric.label for metric in app.metric] == [
                "Model prediction",
                "Estimated bad-credit class probability",
            ]
