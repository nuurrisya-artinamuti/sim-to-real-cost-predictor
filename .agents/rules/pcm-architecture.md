# PCM Architecture & Constraints

* Always write code strictly adhering to PEP 8 standards.
* The Risk Engine utilizes a Random Forest model; absolutely NO Monte Carlo simulation logic should be generated or introduced.
* Ensure the 90% Confidence Interval is calculated by pulling the 5th and 95th percentiles from `self.model.estimators_`.
* When generating the Market Forecaster logic, ensure data ingestion explicitly filters for the years `2020 <= year <= 2024`.
* Ensure synthetic data generation strictly uses PERT/Beta distributions for noise simulation.
* Never generate unverified external links or rely on external APIs for data; use synthetic data generation for testing.
* When testing, write `pytest` compatible unit tests with the `test_` prefix.
