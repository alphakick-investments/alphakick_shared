import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

import pandas as pd
import numpy as np

import yfinance as yf  # will likely need to pip install
# note an issue with yfinance here: github.com/ranaroussi/yfinance/issues/214

from indicators import *
from col_refs import *


def get_price(entity_symbol, data_source='yfinance', table_type='eod_price'):
    ticker = yf.Ticker(entity_symbol)
    pd_price = ticker.history(period='max', auto_adjust=False, rounding=False)

    dic_cols_rename = dic_cols_rename_ref[data_source][table_type]

    if not set(pd_price.columns).issubset(dic_cols_rename.keys()):
        print('WARNING: unknown columns encountered')

    pd_price = pd_price.rename(columns=dic_cols_rename)
    pd_price['entity_symbol'] = entity_symbol
    pd_price = pd_price.drop(['dividend', 'split'], axis=1)
    pd_price.index.name = None

    return pd_price


def get_indicator(pd_prices):

    pd_indicator = calc_indicators(pd_prices, col_high='high', col_low='low', col_close='close', col_volume='volume', bool_fillna=False)

    pd_indicator = pd_indicator.dropna(axis=1, how='all')

    #for c in pd_indicator.columns:
    #    if pd_indicator[c].isnull().sum(axis=0) / len(pd_indicator) > 0.5:
    #        pd_indicator = pd_indicator.drop(c, axis=1)

    pd_indicator = pd_indicator.fillna(method='ffill')
    pd_indicator = pd_indicator.fillna(method='bfill')

    return pd_indicator


def compute_portvals(pd_orders, pd_prices, sv=1000000, comm=9.95, imp=0.005):
    ls_symbols = pd_prices['entity_symbol'].unique()

    sd = pd_orders.index.min()
    ed = pd_orders.index.max()

    pd_prices = pd_prices.loc[sd:ed, :]

    pd_prices_all = pd.DataFrame([], index=pd_prices.index)

    for symb in ls_symbols:
        pd_prices_all[symb] = pd_prices.loc[pd_prices['entity_symbol'] == symb, 'close']

    pd_account = pd.DataFrame(0, index=pd_prices_all.index, columns=['credit', 'debit', 'fees'])
    pd_positions = pd.DataFrame(0, index=pd_prices_all.index, columns=pd_prices_all.columns)

    pd_account.loc[sd, 'credit'] = sv

    for index, row in pd_orders.iterrows():
        symb = row['entity_symbol']
        order = row['order']
        shares = row['shares']

        price = pd_prices_all.loc[index, symb]

        trade_value = price * shares
        trade_cost = comm + imp * trade_value

        if order == 'BUY':
            pd_positions.loc[index, symb] = pd_positions.loc[index, symb] + shares
            pd_account.loc[index, 'debit'] = pd_account.loc[index, 'debit'] - trade_value

        else:
            pd_positions.loc[index, symb] = pd_positions.loc[index, symb] - shares
            pd_account.loc[index, 'credit'] = pd_account.loc[index, 'credit'] + trade_value

        pd_account.loc[index, 'fees'] = pd_account.loc[index, 'fees'] - trade_cost

    pd_positions = pd_positions.cumsum()

    pd_account['balance'] = pd_account[['credit', 'debit', 'fees']].sum(axis=1)
    pd_account['balance'] = pd_account['balance'].cumsum()

    pd_holdings = pd_positions * pd_prices_all
    pd_account['holdings'] = pd_holdings.sum(axis=1)

    pd_account['value'] = pd_account[['balance', 'holdings']].sum(axis=1)

    portvals = pd.DataFrame(pd_account['value'].values, pd_account.index, ['port_val'])

    return portvals


def compute_portfolio_stats(port_val, rfr=0.0, sf=252.0):
    dr = (port_val / port_val.shift(1)) - 1
    cr = (port_val.iloc[-1] / port_val.iloc[0]) - 1

    dr.iloc[0] = 0
    dr = dr[1:]
    dr_rfr = dr - rfr

    adr_rfr = dr_rfr.mean()
    adr = dr.mean()
    sddr = dr.std()

    sr = np.sqrt(sf) * (adr_rfr / sddr)

    return cr, adr, sddr, sr


def market_simulator(pd_orders, pd_benchmark, pd_prices,
                     sv=1000000, comm=9.95, imp=0.005,
                     daily_rf=0.0, samples_per_year=252.0,
                     save_fig=False, gen_stats=True,
                     fig_name='plot.png', stats_name='stats.tsv'):

    portvals = compute_portvals(pd_orders, pd_prices, sv, comm, imp)
    cr_port, adr_port, sddr_port, sr_port = compute_portfolio_stats(portvals, rfr=daily_rf, sf=samples_per_year)

    benchvals = compute_portvals(pd_benchmark, pd_prices, sv, comm, imp)
    cr_bench, adr_bench, sddr_bench, sr_bench = compute_portfolio_stats(benchvals, rfr=daily_rf, sf=samples_per_year)

    pd_stats = pd.DataFrame({'cumulative_return': ['{:0.6f}'.format(cr_port[0]), '{:0.6f}'.format(cr_bench[0])],
                             'average_daily_return': ['{:0.6f}'.format(adr_port[0]), '{:0.6f}'.format(adr_bench[0])],
                             'std_dev_of_returns': ['{:0.6f}'.format(sddr_port[0]), '{:0.6f}'.format(sddr_bench[0])],
                             'sharpe_ratio': ['{:0.4f}'.format(sr_port[0]), '{:0.4f}'.format(sr_bench[0])],
                             'number_of_trades': [len(pd_orders), len(pd_benchmark)]},
                            columns=['cumulative_return', 'average_daily_return', 'std_dev_of_returns', 'sharpe_ratio', 'number_of_trades'],
                            index=['portfolio', 'benchmark'])

    if gen_stats == True:
        pd_stats.to_csv(stats_name, sep='\t')

    portvals_norm = portvals / portvals.iloc[0]
    benchvals_norm = benchvals / benchvals.iloc[0]

    portvals_norm = portvals_norm.reset_index()
    benchvals_norm = benchvals_norm.reset_index()

    vals_norm = pd.merge(portvals_norm, benchvals_norm, on='index', how='outer')
    vals_norm = vals_norm.set_index('index').sort_index().ffill()
    vals_norm.columns = ['portfolio', 'benchmark']

    fig = plt.figure(figsize=(20, 12))
    ax1 = fig.add_subplot(111)

    plt.plot(vals_norm.index, vals_norm['portfolio'], color='black', label='portfolio')
    plt.plot(vals_norm.index, vals_norm['benchmark'], color='blue', label='benchmark')

    for date in pd_orders.index:

        if pd_orders.loc[date, 'order'] == 'BUY':
            plt.axvline(date, color='g', alpha=0.5)

        else:
            plt.axvline(date, color='r', alpha=0.5)

    plt.title('strategy_learner portfolio vs. benchmark')
    plt.ylabel('normalized value')
    plt.legend(loc='upper right')

    if save_fig == True:
        plt.savefig(fig_name, bbox_inches='tight')

    else:
        plt.interactive(True)
        plt.show()