from model.predictor import Predictor


def main():

    predictor = Predictor()

    digit, conf, t = predictor.predict()

    print(f"\nPredicted Digit: {digit}")
    print(f"Confidence: {conf:.4f}")
    print(f"Inference Time: {t:.6f} sec")


if __name__ == "__main__":
    main()