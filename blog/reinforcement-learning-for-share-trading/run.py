import datetime as dt
import pandas as pd

from strategy_learner import *
from indicators import *
from utils import *


def main():
    entity_symbol = "AAPL"
    sv = 100000
    comm = 0.0
    imp = 0.0

    pd_prices = get_price(entity_symbol)
    pd_features = get_indicator(pd_prices)

    slr = StrategyLearner(impact=imp)

    # in-sample
    sd = dt.datetime(2010, 1, 4)
    ed = dt.datetime(2019, 12, 31)

    pd_train_trades = slr.addEvidence(entity_symbol, pd_prices, pd_features,
                                      sd=sd, ed=ed, sv=sv)
    #print(pd_train_trades)

    pd_train_bench = pd.DataFrame(data=[(pd_train_trades.index.min(), entity_symbol, 'BUY', sv / pd_prices.loc[sd, 'adj_close']),
                                        (pd_train_trades.index.max(), entity_symbol, 'SELL', sv / pd_prices.loc[sd, 'adj_close'])],
                                  columns=['date', 'entity_symbol', 'order', 'shares'])

    pd_train_bench = pd_train_bench.set_index('date')

    # out-of-sample
    sd = dt.datetime(2020, 1, 2)
    ed = dt.datetime(2020, 4, 9)

    pd_test_trades = slr.testPolicy(entity_symbol, pd_prices, pd_features,
                                    sd=sd, ed=ed, sv=sv)
    #print(pd_test_trades)

    pd_test_bench = pd.DataFrame(data=[(pd_test_trades.index.min(), entity_symbol, 'BUY', sv / pd_prices.loc[sd, 'adj_close']),
                                       (pd_test_trades.index.max(), entity_symbol, 'SELL', sv / pd_prices.loc[sd, 'adj_close'])],
                                 columns=['date', 'entity_symbol', 'order', 'shares'])

    pd_test_bench = pd_test_bench.set_index('date')

    market_simulator(pd_train_trades, pd_train_bench, pd_prices,
                     sv=sv, comm=comm, imp=imp,
                     save_fig=True, gen_stats=True,
                     fig_name='AAPL_sl_train_plot.png',
                     stats_name='AAPL_sl_train_stats.tsv')

    market_simulator(pd_test_trades, pd_test_bench, pd_prices,
                     sv=sv, comm=comm, imp=imp,
                     save_fig=True, gen_stats=True,
                     fig_name='AAPL_sl_test_plot.png',
                     stats_name='AAPL_sl_test_stats.tsv')


if __name__=="__main__":
    main()