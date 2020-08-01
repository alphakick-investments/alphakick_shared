import seaborn as sns
sns.set_style('darkgrid')

import datetime as dt

import pandas as pd

from q_learner import *
from indicators import *
from col_refs import *


class StrategyLearner(object):

    def __init__(self, impact=0.0, verbose=False):
        self.impact = impact
        self.verbose = verbose

        self.ql = QLearner(num_states=9999, num_actions=3, alpha=0.2, gamma=0.9, rar=0.98, radr=0.999, dyna=0,
                           verbose=False)

    def get_bins(self, ps_feature, num_steps):
        step_size = int(len(ps_feature.index) / (num_steps + 1))
        ps_feature = ps_feature.sort_values()

        bins = []
        for s in range(0, num_steps + 1):
            if s == 0:
                bins.append(ps_feature.iloc[0])

            elif s < num_steps:
                bins.append(ps_feature.iloc[s * step_size])

            else:
                bins.append(ps_feature.iloc[-1])

        return bins


    def get_trade(self, action, holdings, trade_size):

        if (action == 1) and (holdings < trade_size):
            order = 'BUY'

            if holdings == 0:
                shares = trade_size
                holdings += trade_size
            else:
                shares = (2 * trade_size)
                holdings += (2 * trade_size)

        elif (action == 2) and (holdings > -trade_size):
            order = 'SELL'

            if holdings == 0:
                shares = trade_size
                holdings -= trade_size
            else:
                shares = (2 * trade_size)
                holdings -= (2 * trade_size)

        else:
            order = 'HOLD'
            shares = 0

        return order, shares, holdings


    def addEvidence(self, symbol, pd_prices, pd_features,
                    sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=10000):

        pd_prices = pd_prices.loc[sd:ed, :]
        pd_features = pd_features.loc[sd:ed, :]

        # pd_states = pd_features[['volume_adi', 'volume_obv', 'volume_cmf', 'volume_vpt', 'volume_nvi']].copy()
        pd_states = pd_features[['momentum_roc', 'momentum_rsi', 'trend_macd_diff', 'volatility_bbm']].copy()

        num_steps = 9  # note num_states for ql

        for c in pd_states.columns:
            ls_bins = self.get_bins(pd_states[c], num_steps)
            pd_states[c] = pd.cut(pd_states[c], bins=ls_bins, labels=range(0, num_steps)).fillna(0)  # duplicates='drop'

        pd_states['state'] = pd_states.applymap(str).sum(axis=1).astype(int)

        sym_prices = pd_prices['adj_close']
        sym_returns = (sym_prices[1:] / sym_prices[:-1].values) - 1

        ps_symbol = pd.DataFrame(symbol, index=pd_prices.index, columns=['entity_symbol'])
        ps_order = pd.DataFrame('HOLD', index=pd_prices.index, columns=['order'])
        ps_shares = pd.DataFrame(0, index=pd_prices.index, columns=['shares'])

        pd_trades = pd.concat([ps_symbol, ps_order, ps_shares], axis=1)
        pd_trades.columns = ['entity_symbol', 'order', 'shares']

        initial_state = pd_states['state'].iloc[0]
        self.ql.querysetstate(initial_state)

        trade_size = sv / pd_prices['adj_close'].iloc[0]

        pd_trades_copy = pd_trades.copy()

        i = 0
        j = 0

        min_epoch = 20
        cov_epoch = 40
        max_epoch = 500

        while i < max_epoch:

            i += 1
            holdings = 0

            if pd_trades.equals(pd_trades_copy):
                if i > min_epoch:
                    j += 1

                if j > cov_epoch:
                    break

            pd_trades_copy = pd_trades.copy()

            for index, row in pd_prices[1:].iterrows():
                state = pd_states.loc[index, 'state']
                reward = holdings * sym_returns.loc[index] * (1 - self.impact)
                action = self.ql.query(state, reward)

                order, shares, holdings = self.get_trade(action, holdings, trade_size)

                ps_order.loc[index]['order'] = order
                ps_shares.loc[index]['shares'] = shares

            pd_trades = pd.concat([ps_symbol, ps_order, ps_shares], axis=1)
            pd_trades.columns = ['entity_symbol', 'order', 'shares']

        pd_trades = pd_trades.loc[pd_trades['shares'] != 0, :]

        return pd_trades


    def testPolicy(self, symbol, pd_prices, pd_features,
                   sd=dt.datetime(2010, 1, 1), ed=dt.datetime(2011, 12, 31), sv=10000):

        pd_prices = pd_prices.loc[sd:ed]
        pd_features = pd_features.loc[sd:ed]

        # pd_states = pd_features[['volume_adi', 'volume_obv', 'volume_cmf', 'volume_vpt', 'volume_nvi']].copy()
        pd_states = pd_features[['momentum_roc', 'momentum_rsi', 'trend_macd', 'volatility_bbm']].copy()

        num_steps = 5

        for c in pd_states.columns:
            ls_bins = self.get_bins(pd_states[c], num_steps)
            pd_states[c] = pd.cut(pd_states[c], bins=ls_bins, labels=range(0, num_steps)).fillna(0)  # duplicates='drop'

        pd_states['state'] = pd_states.applymap(str).sum(axis=1).astype(int)

        sym_prices = pd_prices['adj_close']
        sym_returns = (sym_prices[1:] / sym_prices[:-1].values) - 1

        ps_symbol = pd.DataFrame(symbol, index=pd_prices.index, columns=['entity_symbol'])
        ps_order = pd.DataFrame('HOLD', index=pd_prices.index, columns=['order'])
        ps_shares = pd.DataFrame(0, index=pd_prices.index, columns=['shares'])

        pd_trades = pd.concat([ps_symbol, ps_order, ps_shares], axis=1)
        pd_trades.columns = ['entity_symbol', 'order', 'shares']

        initial_state = pd_states['state'].iloc[0]
        self.ql.querysetstate(initial_state)

        trade_size = sv / pd_prices['adj_close'].iloc[0]

        holdings = 0

        for index, row in pd_prices[1:].iterrows():
            state = pd_states.loc[index, 'state']
            reward = holdings * sym_returns.loc[index] * (1 - self.impact)
            action = self.ql.query(state, reward)

            order, shares, holdings = self.get_trade(action, holdings, trade_size)

            ps_order.loc[index]['order'] = order
            ps_shares.loc[index]['shares'] = shares

            pd_trades = pd.concat([ps_symbol, ps_order, ps_shares], axis=1)
            pd_trades.columns = ['entity_symbol', 'order', 'shares']

        pd_trades = pd_trades.loc[pd_trades['shares'] != 0, :]

        return pd_trades