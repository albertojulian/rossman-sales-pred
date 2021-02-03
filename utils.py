import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def metric(preds, actuals):
    preds = preds.reshape(-1)
    actuals = actuals.reshape(-1)
    assert preds.shape == actuals.shape
    return 100 * np.linalg.norm((actuals - preds) / actuals) / np.sqrt(preds.shape[0])

# plot the sales for [n_rows x n_cols] stores, for some date points
# plot the mean sales per month for [n_rows x n_cols] stores
def plot_stores_sales(stores_df, n_rows=5, n_cols=5, store_batch=0, plot_type="raw", n_date_points=50, ref_line=None):
    # plot_type:
    #   "raw" plots sales for n_date_points, for [n_rows x n_cols] stores
    #   "monthly_mean" plots mean sales per month, for [n_rows x n_cols] stores
    first_store = store_batch * n_rows * n_cols
    fig, axs = plt.subplots(n_rows, n_cols)  # , sharex=True)
    plt.xticks(rotation=45)  # just applies to last plot
    store_ids = stores_df.Store.unique()
    for row in range(n_rows):
        for col in range(n_cols):
            idx = first_store + row * n_cols + col
            store_id = store_ids[idx]
            if idx < store_ids.shape[0]:
                store_sales_df = stores_df[stores_df["Store"] == store_id]
                if plot_type=="raw":
                    n_date_points = min(n_date_points, store_sales_df.shape[0])
                    # axs[row, col].plot(store_sales_df.loc[:50, 'Date'], store_sales_df.loc[:50, "Sales"]) # does not work
                    axs[row, col].plot(store_sales_df['Date'][:n_date_points], store_sales_df["Sales"][:n_date_points])
                elif plot_type=="monthly_mean":
                    axs[row, col].plot(store_sales_df['Sales'].groupby(store_sales_df.Date.dt.month).mean())
                store_txt = f"Store ID ={store_id}"
                axs[row, col].text(0, 1, str(store_txt), horizontalalignment='left', verticalalignment='top',
                                transform = axs[row, col].transAxes)

                if type(ref_line) is int or type(ref_line) is float:
                    axs[row, col].axhline(ref_line)
                elif type(ref_line) is pd.DataFrame:
                    axs[row, col].axhline(ref_line.loc[store_id, "Sales"])

    plt.show()

if __name__ == "__main__":
    data_dir = "data"
    test_filename = "holdout.csv"
    test_filepath = os.path.join(data_dir, test_filename)
    test_df = pd.read_csv(test_filepath, dtype={'StateHoliday': 'string'})
    test_df.loc[:, 'Date'] = pd.to_datetime(test_df.loc[:, 'Date'])
    test_df = test_df[test_df["Sales"] > 0]

    n_rows = 10
    n_cols = 5
    store_batch = 0 # 1, 2, ...
    n_date_points = 100
    baseline = 6800
    # plot_stores_sales(test_df, n_rows=n_rows, n_cols=n_cols, store_batch=store_batch, plot_type="monthly_mean", n_date_points=n_date_points, ref_line=baseline)
    # for store_batch in range(6):
    plot_stores_sales(test_df, n_rows=n_rows, n_cols=n_cols, store_batch=store_batch, plot_type="monthly_mean", ref_line=baseline)
