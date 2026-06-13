import csv
import joblib

from personal_style_pl.cli import main
from personal_style_pl.models.supervised import SupervisedStyleModel


def test_train_supervised_produces_model(tmp_path):
    path = tmp_path / "contrast.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["text", "label", "source"])
        for i in range(4):
            w.writerow([f"Mój prosty tekst osobisty numer {i}. Krótko i zwięźle, bez ozdób.",
                        "mine", f"me{i}"])
            w.writerow([f"Warto zauważyć, że w dzisiejszych czasach kluczowe znaczenie ma rozwój {i}.",
                        "other", f"ot{i}"])
    out = tmp_path / "model.joblib"
    rc = main(["train-supervised", "--csv", str(path), "--text-col", "text",
               "--label-col", "label", "--output", str(out)])
    assert rc == 0 and out.exists()
    model = joblib.load(out)
    assert isinstance(model, SupervisedStyleModel)
    proba = model.predict_proba_mine(["Mój prosty tekst osobisty. Krótko."])
    assert 0.0 <= float(proba[0]) <= 1.0
