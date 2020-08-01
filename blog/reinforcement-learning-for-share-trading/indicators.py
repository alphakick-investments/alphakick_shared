import numpy as np

from ta.momentum import (AwesomeOscillatorIndicator, KAMAIndicator,
                         ROCIndicator, RSIIndicator, StochasticOscillator,
                         TSIIndicator, UltimateOscillator, WilliamsRIndicator)

from ta.others import (CumulativeReturnIndicator, DailyLogReturnIndicator,
                       DailyReturnIndicator)

from ta.trend import (MACD, ADXIndicator, AroonIndicator, CCIIndicator,
                      DPOIndicator, EMAIndicator, IchimokuIndicator,
                      KSTIndicator, MassIndex, PSARIndicator, SMAIndicator,
                      TRIXIndicator, VortexIndicator)

from ta.volatility import (AverageTrueRange, BollingerBands, DonchianChannel,
                           KeltnerChannel)

from ta.volume import (AccDistIndexIndicator, ChaikinMoneyFlowIndicator,
                       EaseOfMovementIndicator, ForceIndexIndicator,
                       MFIIndicator, NegativeVolumeIndexIndicator,
                       OnBalanceVolumeIndicator, VolumePriceTrendIndicator)


def calc_indicators(df_prices, col_high='high', col_low='low', col_close='adj_close', col_volume='volume', bool_fillna=False):
    df = df_prices.copy()

    # Accumulation Distribution Index
    try:
        df['volume_adi'] = AccDistIndexIndicator(
            high=df[col_high], low=df[col_low], close=df[col_close], volume=df[col_volume], fillna=bool_fillna).acc_dist_index()
    except Exception:
        print('ERROR: AccDistIndexIndicator calc issue')
        df['volume_adi'] = np.nan
        pass

    # On Balance Volume
    try:
        df['volume_obv'] = OnBalanceVolumeIndicator(
            close=df[col_close], volume=df[col_volume], fillna=bool_fillna).on_balance_volume()
    except Exception:
        print('ERROR: OnBalanceVolumeIndicator calc issue')
        df['volume_obv'] = np.nan
        pass
    
    # Chaikin Money Flow
    try:
        df['volume_cmf'] = ChaikinMoneyFlowIndicator(
            high=df[col_high], low=df[col_low], close=df[col_close], volume=df[col_volume], fillna=bool_fillna).chaikin_money_flow()
    except Exception:
        print('ERROR: ChaikinMoneyFlowIndicator calc issue')
        df['volume_cmf'] = np.nan
        pass

    # Force Index
    try:
        df['volume_fi'] = ForceIndexIndicator(
            close=df[col_close], volume=df[col_volume], n=13, fillna=bool_fillna).force_index()
    except Exception:
        print('ERROR: ForceIndexIndicator calc issue')
        df['volume_fi'] = np.nan
        pass

    # Money Flow Indicator
    try:
        df['momentum_mfi'] = MFIIndicator(
            high=df[col_high], low=df[col_low], close=df[col_close], volume=df[col_volume], n=14, fillna=bool_fillna).money_flow_index()
    except Exception:
        print('ERROR: MFIIndicator calc issue')
        df['momentum_mfi'] = np.nan
        pass

    # Ease of Movement
    try:
        indicator = EaseOfMovementIndicator(high=df[col_high], low=df[col_low], volume=df[col_volume], n=14, fillna=bool_fillna)
        df['volume_em'] = indicator.ease_of_movement()
        df['volume_sma_em'] = indicator.sma_ease_of_movement()
    except Exception:
        print('ERROR: EaseOfMovementIndicator calc issue')
        df['volume_em'] = np.nan
        df['volume_sma_em'] = np.nan
        pass

    # Volume Price Trend
    try:
        df['volume_vpt'] = VolumePriceTrendIndicator(
            close=df[col_close], volume=df[col_volume], fillna=bool_fillna).volume_price_trend()
    except Exception:
        print('ERROR: VolumePriceTrendIndicator calc issue')
        df['volume_vpt'] = np.nan
        pass

    # Negative Volume Index
    try:
        df['volume_nvi'] = NegativeVolumeIndexIndicator(
            close=df[col_close], volume=df[col_volume], fillna=bool_fillna).negative_volume_index()
    except Exception:
        print('ERROR: NegativeVolumeIndexIndicator calc issue')
        df['volume_nvi'] = np.nan
        pass

    # Average True Range
    try:
        df['volatility_atr'] = AverageTrueRange(
            close=df[col_close], high=df[col_high], low=df[col_low], n=10, fillna=bool_fillna).average_true_range()
    except Exception:
        print('ERROR: AverageTrueRange calc issue')
        df['volatility_atr'] = np.nan
        pass

    # Bollinger Bands
    try:
        indicator_bb = BollingerBands(close=df[col_close], n=20, ndev=2, fillna=bool_fillna)
        df['volatility_bbm'] = indicator_bb.bollinger_mavg()
        df['volatility_bbh'] = indicator_bb.bollinger_hband()
        df['volatility_bbl'] = indicator_bb.bollinger_lband()
        df['volatility_bbw'] = indicator_bb.bollinger_wband()
        df['volatility_bbp'] = indicator_bb.bollinger_pband()
        df['volatility_bbhi'] = indicator_bb.bollinger_hband_indicator()
        df['volatility_bbli'] = indicator_bb.bollinger_lband_indicator()
    except Exception:
        print('ERROR: BollingerBands calc issue')
        df['volatility_bbm'] = np.nan
        df['volatility_bbh'] = np.nan
        df['volatility_bbl'] = np.nan
        df['volatility_bbw'] = np.nan
        df['volatility_bbp'] = np.nan
        df['volatility_bbhi'] = np.nan
        df['volatility_bbli'] = np.nan
        pass

    # Keltner Channel
    try:
        indicator_kc = KeltnerChannel(close=df[col_close], high=df[col_high], low=df[col_low], n=10, fillna=bool_fillna)
        df['volatility_kcc'] = indicator_kc.keltner_channel_mband()
        df['volatility_kch'] = indicator_kc.keltner_channel_hband()
        df['volatility_kcl'] = indicator_kc.keltner_channel_lband()
        df['volatility_kcw'] = indicator_kc.keltner_channel_wband()
        df['volatility_kcp'] = indicator_kc.keltner_channel_pband()
        df['volatility_kchi'] = indicator_kc.keltner_channel_hband_indicator()
        df['volatility_kcli'] = indicator_kc.keltner_channel_lband_indicator()
    except Exception:
        print('ERROR: KeltnerChannel calc issue')
        df['volatility_kcc'] = np.nan
        df['volatility_kch'] = np.nan
        df['volatility_kcl'] = np.nan
        df['volatility_kcw'] = np.nan
        df['volatility_kcp'] = np.nan
        df['volatility_kchi'] = np.nan
        df['volatility_kcli'] = np.nan
        pass

    # Donchian Channel
    try:
        indicator_dc = DonchianChannel(close=df[col_close], n=20, fillna=bool_fillna)
        df['volatility_dcl'] = indicator_dc.donchian_channel_lband()
        df['volatility_dch'] = indicator_dc.donchian_channel_hband()
        df['volatility_dchi'] = indicator_dc.donchian_channel_hband_indicator()
        df['volatility_dcli'] = indicator_dc.donchian_channel_lband_indicator()
    except Exception:
        print('ERROR: KeltnerChannel calc issue')
        df['volatility_dcl'] = np.nan
        df['volatility_dch'] = np.nan
        df['volatility_dchi'] = np.nan
        df['volatility_dcli'] = np.nan
        pass

    # MACD
    try:
        indicator_macd = MACD(close=df[col_close], n_slow=26, n_fast=12, n_sign=9, fillna=bool_fillna)
        df['trend_macd'] = indicator_macd.macd()
        df['trend_macd_signal'] = indicator_macd.macd_signal()
        df['trend_macd_diff'] = indicator_macd.macd_diff()
    except Exception:
        print('ERROR: MACD calc issue')
        df['trend_macd'] = np.nan
        df['trend_macd_signal'] = np.nan
        df['trend_macd_diff'] = np.nan
        pass
    
    # SMAs
    try:
        df['trend_sma_fast'] = SMAIndicator(
            close=df[col_close], n=12, fillna=bool_fillna).sma_indicator()
        df['trend_sma_slow'] = SMAIndicator(
            close=df[col_close], n=26, fillna=bool_fillna).sma_indicator()
    except Exception:
        print('ERROR: SMAIndicator calc issue')
        df['trend_sma_fast'] = np.nan
        df['trend_sma_slow'] = np.nan
        pass

    # EMAs
    try:
        df['trend_ema_fast'] = EMAIndicator(
            close=df[col_close], n=12, fillna=bool_fillna).ema_indicator()
        df['trend_ema_slow'] = EMAIndicator(
            close=df[col_close], n=26, fillna=bool_fillna).ema_indicator()
    except Exception:
        print('ERROR: EMAIndicator calc issue')
        df['trend_ema_fast'] = np.nan
        df['trend_ema_slow'] = np.nan
        pass

    # Average Directional Movement Index (ADX)
    try:
        indicator = ADXIndicator(high=df[col_high], low=df[col_low], close=df[col_close], n=14, fillna=bool_fillna)
        df['trend_adx'] = indicator.adx()
        df['trend_adx_pos'] = indicator.adx_pos()
        df['trend_adx_neg'] = indicator.adx_neg()
    except Exception:
        print('ERROR: ADXIndicator calc issue')
        df['trend_adx'] = np.nan
        df['trend_adx_pos'] = np.nan
        df['trend_adx_neg'] = np.nan
        pass

    # Vortex Indicator
    try:
        indicator = VortexIndicator(high=df[col_high], low=df[col_low], close=df[col_close], n=14, fillna=bool_fillna)
        df['trend_vortex_ind_pos'] = indicator.vortex_indicator_pos()
        df['trend_vortex_ind_neg'] = indicator.vortex_indicator_neg()
        df['trend_vortex_ind_diff'] = indicator.vortex_indicator_diff()
    except Exception:
        print('ERROR: VortexIndicator calc issue')
        df['trend_vortex_ind_pos'] = np.nan
        df['trend_vortex_ind_neg'] = np.nan
        df['trend_vortex_ind_diff'] = np.nan
        pass

    # TRIX Indicator
    try:
        indicator = TRIXIndicator(close=df[col_close], n=15, fillna=bool_fillna)
        df['trend_trix'] = indicator.trix()
    except Exception:
        print('ERROR: TRIXIndicator calc issue')
        df['trend_trix'] = np.nan
        pass
    
    # Mass Index
    try:
        indicator = MassIndex(high=df[col_high], low=df[col_low], n=9, n2=25, fillna=bool_fillna)
        df['trend_mass_index'] = indicator.mass_index()
    except Exception:
        print('ERROR: MassIndex calc issue')
        df['trend_mass_index'] = np.nan
        pass
    
    # CCI Indicator
    try:
        indicator = CCIIndicator(high=df[col_high], low=df[col_low], close=df[col_close], n=20, c=0.015, fillna=bool_fillna)
        df['trend_cci'] = indicator.cci()
    except Exception:
        print('ERROR: CCIIndicator calc issue')
        df['trend_cci'] = np.nan
        pass
    
    # DPO Indicator
    try:
        indicator = DPOIndicator(close=df[col_close], n=20, fillna=bool_fillna)
        df['trend_dpo'] = indicator.dpo()
    except Exception:
        print('ERROR: DPOIndicator calc issue')
        df['trend_dpo'] = np.nan
        pass
    
    # KST Indicator
    try:
        indicator = KSTIndicator(close=df[col_close],
                                 r1=10, r2=15, r3=20,
                                 r4=30, n1=10, n2=10, n3=10,
                                 n4=15, nsig=9, fillna=bool_fillna)
        df['trend_kst'] = indicator.kst()
        df['trend_kst_sig'] = indicator.kst_sig()
        df['trend_kst_diff'] = indicator.kst_diff()
    except Exception:
        print('ERROR: KSTIndicator calc issue')
        df['trend_kst'] = np.nan
        df['trend_kst_sig'] = np.nan
        df['trend_kst_diff'] = np.nan
        pass
    
    # Ichimoku Indicator
    try:
        indicator = IchimokuIndicator(high=df[col_high], low=df[col_low], n1=9, n2=26, n3=52, visual=False, fillna=bool_fillna)
        df['trend_ichimoku_a'] = indicator.ichimoku_a()
        df['trend_ichimoku_b'] = indicator.ichimoku_b()
        indicator = IchimokuIndicator(high=df[col_high], low=df[col_low], n1=9, n2=26, n3=52, visual=True, fillna=bool_fillna)
        df['trend_visual_ichimoku_a'] = indicator.ichimoku_a()
        df['trend_visual_ichimoku_b'] = indicator.ichimoku_b()
    except Exception:
        print('ERROR: IchimokuIndicator calc issue')
        df['trend_ichimoku_a'] = np.nan
        df['trend_ichimoku_b'] = np.nan
        df['trend_visual_ichimoku_a'] = np.nan
        df['trend_visual_ichimoku_b'] = np.nan
        pass

    # Aroon Indicator
    try:
        indicator = AroonIndicator(close=df[col_close], n=25, fillna=bool_fillna)
        df['trend_aroon_up'] = indicator.aroon_up()
        df['trend_aroon_down'] = indicator.aroon_down()
        df['trend_aroon_ind'] = indicator.aroon_indicator()
    except Exception:
        print('ERROR: AroonIndicator calc issue')
        df['trend_aroon_up'] = np.nan
        df['trend_aroon_down'] = np.nan
        df['trend_aroon_ind'] = np.nan
        pass

    # PSAR Indicator
    try:
        indicator = PSARIndicator(high=df[col_high], low=df[col_low], close=df[col_close], step=0.02, max_step=0.20, fillna=bool_fillna)
        df['trend_psar'] = indicator.psar()
        df['trend_psar_up'] = indicator.psar_up()
        df['trend_psar_down'] = indicator.psar_down()
        df['trend_psar_up_indicator'] = indicator.psar_up_indicator()
        df['trend_psar_down_indicator'] = indicator.psar_down_indicator()
    except Exception:
        print('ERROR: PSARIndicator calc issue')
        df['trend_psar'] = np.nan
        df['trend_psar_up'] = np.nan
        df['trend_psar_down'] = np.nan
        df['trend_psar_up_indicator'] = np.nan
        df['trend_psar_down_indicator'] = np.nan
        pass

    # Relative Strength Index (RSI)
    try:
        df['momentum_rsi'] = RSIIndicator(close=df[col_close], n=14, fillna=bool_fillna).rsi()
    except Exception:
        print('ERROR: RSIIndicator calc issue')
        df['momentum_rsi'] = np.nan
        pass

    # TSI Indicator
    try:
        df['momentum_tsi'] = TSIIndicator(close=df[col_close], r=25, s=13, fillna=bool_fillna).tsi()
    except Exception:
        print('ERROR: TSIIndicator calc issue')
        df['momentum_tsi'] = np.nan
        pass
    
    # Ultimate Oscillator
    try:
        df['momentum_uo'] = UltimateOscillator(
            high=df[col_high], low=df[col_low], close=df[col_close], s=7, m=14, len=28, ws=4.0, wm=2.0, wl=1.0,
            fillna=bool_fillna).uo()
    except Exception:
        print('ERROR: UltimateOscillator calc issue')
        df['momentum_uo'] = np.nan
        pass
    
    # Stoch Indicator
    try:
        indicator = StochasticOscillator(high=df[col_high], low=df[col_low], close=df[col_close], n=14, d_n=3, fillna=bool_fillna)
        df['momentum_stoch'] = indicator.stoch()
        df['momentum_stoch_signal'] = indicator.stoch_signal()
    except Exception:
        print('ERROR: StochasticOscillator calc issue')
        df['momentum_stoch'] = np.nan
        df['momentum_stoch_signal'] = np.nan
        pass

    # Williams R Indicator
    try:
        df['momentum_wr'] = WilliamsRIndicator(
            high=df[col_high], low=df[col_low], close=df[col_close], lbp=14, fillna=bool_fillna).wr()
    except Exception:
        print('ERROR: WilliamsRIndicator calc issue')
        df['momentum_wr'] = np.nan
        pass
    
    # Awesome Oscillator
    try:
        df['momentum_ao'] = AwesomeOscillatorIndicator(
            high=df[col_high], low=df[col_low], s=5, len=34, fillna=bool_fillna).ao()
    except Exception:
        print('ERROR: AwesomeOscillatorIndicator calc issue')
        df['momentum_ao'] = np.nan
        pass

    # KAMA
    try:
        df['momentum_kama'] = KAMAIndicator(close=df[col_close], n=10, pow1=2, pow2=30, fillna=bool_fillna).kama()
    except Exception:
        print('ERROR: KAMAIndicator calc issue')
        df['momentum_kama'] = np.nan
        pass

    # Rate Of Change
    try:
        df['momentum_roc'] = ROCIndicator(close=df[col_close], n=12, fillna=bool_fillna).roc()
    except Exception:
        print('ERROR: ROCIndicator calc issue')
        df['momentum_roc'] = np.nan
        pass

    # Daily Return
    try:
        df['others_dr'] = DailyReturnIndicator(close=df[col_close], fillna=bool_fillna).daily_return()
    except Exception:
        print('ERROR: DailyReturnIndicator calc issue')
        df['others_dr'] = np.nan
        pass

    # Daily Log Return
    try:
        df['others_dlr'] = DailyLogReturnIndicator(close=df[col_close], fillna=bool_fillna).daily_log_return()
    except Exception:
        print('ERROR: DailyLogReturnIndicator calc issue')
        df['others_dlr'] = np.nan
        pass

    # Cumulative Return
    try:
        df['others_cr'] = CumulativeReturnIndicator(close=df[col_close], fillna=bool_fillna).cumulative_return()
    except Exception:
        print('ERROR: CumulativeReturnIndicator calc issue')
        df['others_cr'] = np.nan
        pass

    return df