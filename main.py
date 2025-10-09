"""
Main module to run the sustainable heart rate analysis tool.

"""

from src import (
    data_loader,
    preprocessing,
    modelling,
    evaluation,
    evaluation_visuals,
    utils,
)

def main():
    # 1. Load data
    df = data_loader.load_data("data/dummy_running_data.csv")
    utils.check_dataframe(df, "Raw Data")

    # 2. Preprocess
    df_prepared = preprocessing.preprocess_data(df)
    utils.check_dataframe(df_prepared, "Preprocessed Data")

    # 3. Train model
    model, predicted = modelling.fit_sustainable_hr_model(df_prepared)

    # 4. Evaluate performance
    metrics = evaluation.evaluate_model(predicted)
    print("\nModel Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.3f}")

    # 5. Visual diagnostics
    evaluation_visuals.plot_residuals_distribution(
        predicted["avg_hr"], predicted["predicted_sustainable_hr"]
    )
    evaluation_visuals.plot_residuals_vs_feature(predicted, feature="distance_km")

if __name__ == "__main__":
    main()
