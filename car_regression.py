"""Run python car_regression.py in the assignment folder. Requires numpy and Pillow."""
import csv
import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FOLDER = Path(__file__).resolve().parent
FEATURES = ["Mileage_K", "Age_Years", "Horsepower", "Accident", "Num_Owners"]
PREP = 1500


def read_csv(name):
    with (FOLDER / name).open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def features(car):
    return [1] + [({"yes": 1, "no": 0}[car[key]] if key == "Accident"
                   else float(car[key])) for key in FEATURES]


def buy_or_pass(car, auction_price, coefficients):
    estimate = np.dot(features(car), coefficients)
    return "buy" if estimate > auction_price + PREP else "pass"


def save_rows(name, rows):
    with (FOLDER / name).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def p_values(t_scores, df):
    # Two-sided Student t tail. Substitute t = sqrt(df)*tan(theta)
    # in the density, then integrate cos(theta)**(df-1) by Gaussian quadrature.
    nodes, weights = np.polynomial.legendre.leggauss(96)
    def integral(start):
        half = (math.pi / 2 - start) / 2
        theta = start + half * (1 + nodes)
        return half * (weights @ (np.cos(theta) ** (df - 1)))
    return [integral(math.atan(abs(t) / math.sqrt(df))) / integral(0) for t in t_scores]


def plot_prices(actual, predicted, baseline):
    img = Image.new("RGB", (1000, 780), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=20)
    def label(x, y, text, anchor="la", size=None):
        draw.text((x, y), text, fill="#173047", font=ImageFont.load_default(size=size) if size else font, anchor=anchor)
    label(60, 28, "Used car resale prices on the held-out test set", size=31)
    label(60, 80, "Predicted price ($)")
    top = max(10000, math.ceil(max(max(actual), max(predicted), baseline) / 10000) * 10000)
    x = lambda value: 120 + 570 * value / top
    y = lambda value: 680 - 570 * value / top
    for tick in range(0, top + 1, 10000):
        draw.line((x(tick), 110, x(tick), 680), fill="#e1e5e8")
        draw.line((120, y(tick), 690, y(tick)), fill="#e1e5e8")
        label(x(tick), 694, f"{tick//1000}k", "ma")
        label(108, y(tick), f"{tick//1000}k", "rm")
    draw.line((120,680,690,110), fill="#666666", width=2)
    draw.line((120,y(baseline),690,y(baseline)), fill="#c7772a", width=3)
    for a, p in zip(actual, predicted):
        draw.ellipse((x(a)-5,y(p)-5,x(a)+5,y(p)+5), fill="#167a78")
    label(405, 735, "Actual resale price ($)", "ma")
    for i, (color, name) in enumerate([("#167a78","Model estimate"),("#666666","Perfect estimate"),("#c7772a","Baseline estimate")]):
        yy = 175 + i * 42
        draw.line((725, yy, 748, yy), fill=color, width=4)
        label(760, yy, name, "lm")
    img.save(FOLDER / "test_prices.png")


def main():
    sales = read_csv("used_car_sales.csv")
    lots = read_csv("auction_lots.csv")
    X = np.array([features(car) for car in sales])
    y = np.array([float(car["Sale_Price"]) for car in sales])
    order = np.random.default_rng(42).permutation(len(sales))
    n = round(0.7 * len(sales))
    train, test = order[:n], order[n:]
    A, b = X[train], y[train]
    coef, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    assert rank == A.shape[1]
    df = n - rank
    sse = np.sum((b - A @ coef)**2)
    se = np.sqrt(np.diag(np.linalg.inv(A.T @ A)) * sse / df)
    avg = b.mean()
    print("Training / test:", n, len(test), "| baseline price:", avg)
    print("Training adjusted R2:", 1 - (sse/df) / b.var(ddof=1))
    for key, estimate, p in zip(["Intercept"] + FEATURES, coef, p_values(coef/se, df)):
        print(f"{key:15s} coefficient={estimate:10.3f}  p={p:.5g}")
    observed = y[test]
    predictions = X[test] @ coef
    for name, estimates in [("Baseline", avg), ("Regression", predictions)]:
        err = observed - estimates
        print(name, "test R2", 1 - np.sum(err**2)/np.sum((observed-observed.mean())**2),
              "RMSE", np.sqrt(np.mean(err**2)))
    plot_prices(observed, predictions, avg)
    details = []
    for car in lots:
        price = float(car["Auction_Price"])
        model = buy_or_pass(car, price, coef)
        base = "buy" if avg > price + PREP else "pass"
        net = float(car["Later_Sale_Price"]) - price - PREP
        details.append(dict(lot=car["Lot"], auction_price=price, later_sale_price=car["Later_Sale_Price"],
                            model_estimate=round(np.dot(features(car),coef),2), baseline_estimate=round(avg,2),
                            baseline=base, model=model, baseline_profit=net if base=="buy" else 0,
                            model_profit=net if model=="buy" else 0))
    save_rows("auction_decisions.csv", details)
    for rule in ["baseline", "model"]:
        profits = [r[rule+"_profit"] for r in details if r[rule]=="buy"]
        print(rule, "bought", len(profits), "losing purchases", sum(v<0 for v in profits), "profit", sum(profits))
    print("Incremental profit:", sum(r["model_profit"]-r["baseline_profit"] for r in details))


if __name__ == "__main__":
    main()
