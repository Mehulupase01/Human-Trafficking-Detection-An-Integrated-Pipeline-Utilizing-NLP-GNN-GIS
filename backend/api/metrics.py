from backend.metrics.metrics_engine import compute_basic_metrics, plot_year_histogram

def run_metrics_pipeline(df):
    total, nationality_dist, gender_dist, years = compute_basic_metrics(df)
    chart = plot_year_histogram(years)
    return total, nationality_dist, gender_dist, chart