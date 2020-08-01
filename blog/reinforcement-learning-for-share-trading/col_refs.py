dic_cols_rename_yfinance_price_data = {
    'Date': 'date',
    'Symbol': 'entity_symbol',
    'Open': 'open',
    'High': 'high',
    'Low': 'low',
    'Close': 'close',
    'Adj Close': 'adj_close',
    'Volume': 'volume',
    'Dividends': 'dividend',
    'Stock Splits': 'split'
}


dic_cols_rename_yfinance_data_ref = {
    'eod_price': dic_cols_rename_yfinance_price_data,
    'intra_price': dic_cols_rename_yfinance_price_data
}


dic_cols_rename_ref = {
    'yfinance': dic_cols_rename_yfinance_data_ref
}


############################################################

ls_cols_yfinance_price_data = [
    'pr_key',
    'date',
    'entity_symbol',
    'composite_symbol',
    'open',
    'high',
    'low',
    'close',
    'adj_close',
    'volume',
    'dividend',
    'split',
    'last_updated'
]


dic_ls_cols_yfinance_data_ref = {
    'eod_price': ls_cols_yfinance_price_data,
    'intra_price': ls_cols_yfinance_price_data
}


dic_ls_cols_data_ref = {
    'yfinance': dic_ls_cols_yfinance_data_ref
}