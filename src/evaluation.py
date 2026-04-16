if __name__ == "__main__":
    
    try:
        data_path = sys.argv[1] if len(sys.argv) > 1 else "data/processed/train.csv"
        if not os.path.exists(data_path):
            raise FileNotFoundError
    except (IndexError, FileNotFoundError):
        data_path = 'https://raw.githubusercontent.com/raphi-l/my-portfolio/refs/heads/main/datasets/mal_nut_train_sample.csv'
    
    df = load_data(data_path)  
    metrics = train_model(df)

    with open("configs/model_params.yaml") as f:
        model_config = yaml.safe_load(f)

    quality_config = model_config["model_quality"]

    # Exit with error if thresholds not met
    if metrics["accuracy"] < quality_config["min_accuracy"]:
        print(f"\nFAILED: Accuracy below threshold")
        sys.exit(1)
    if metrics["f1_score"] < quality_config["min_f1"]:
        print(f"\nFAILED: F1 score below threshold")
        sys.exit(1)

    print("\nAll thresholds passed!")

    save_model