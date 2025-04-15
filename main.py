import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

#Notes：
#所有文件（未特别说明全部下载即可）：
#   (1)个股月收益率、(2)个股日收益率、(3)综合市场日度交易数据、(4)无风险利率日度、(5)三因子收益率日度、(6)五因子收益率日度、
#   (7)公司研究-治理结构-公司基本情况文件、(8)公司研究-治理结构-治理综合信息文件[(必须有)公告日期+员工人数(尽量不下载公司沿革)]、
#   (9)股票市场交易-基本数据-公司文件、(10)公司研究系列-年中季报公布日期-基本情况文件、
#   (11)资产负债表|(12)利润表|(13)现金流量表(直接法)|(14)现金流量表(间接法)。
#   (15)CH_4_fac_daily_update_20211231中国版四因子文件，免费下载：https://finance.wharton.upenn.edu/~stambaug/
#至少是三年720个交易日数据（rollingols滚动窗口是720）。
#！！！！！注意公司文件、公司基本情况文件中的数据每个公司只有一条且可能在所选范围之外，这两个文件单独下载，扩大选择全部时间段。

#待解决报错：
#【已解决】全部A股数据，计算成功第39(耗时40min)，raw_data.Annodt = raw_data.Annodt.apply(lambda x: x[0:4] + x[5:7] + x[8:10])。TypeError: 'float' object is not subscriptable
#可以选择剔除掉公告日期为缺失值的股票数据
#【待解决】pandas.errors.ParserError: Error tokenizing data. C error: Expected 23 fields in line 2032, saw 27
#time = (int(result.index[i][0:4])-int(listdate.iloc[j,0][0:4]))*12 + int(result.index[i][4:6])-int(listdate.iloc[j,0][4:6])
#age因子计算报错，同时数据文件格式错位需要手工调整，只有这一个因子用到了CG_CO文件



##############################################################################
# begin class


class AShareMarket:

    def __init__(self, mode, local_data_path=None):
        
        if mode == 'local':
            self.mode = 'local'
            self.local_data_path = local_data_path
            self.ST_PT = pd.read_csv(self.local_data_path + 'TRD_Dalyr.csv')
            self.ST_PT.sort_values(by=['Trddt'], inplace=True, ignore_index=True)
            self.ST_PT['Trdmnt'] = list(self.ST_PT['Trddt'])
            self.ST_PT.Trdmnt = self.ST_PT.Trdmnt.apply(lambda x: x[0:4] + x[5:7])#日期标准化
            self.ST_PT.drop_duplicates(subset=['Trdmnt', 'Stkcd'], keep='last', inplace=True)
            self.ST_PT = self.ST_PT[['Stkcd', 'Trdmnt', 'Trdsta']]
            self.monthly_ret = self.get_data('TRD_Mnth', 'Mretwd')

        else:
            self.mode = 'online'
    '''
    Markettype [Market Type] - 1=SSE A share(excluding SSE STAR Market); 
    2= SSE B share; 4= SZSE A share(excluding ChiNext);
    8= SZSE B share; 16= ChiNext; 32=SSE STAR Market.




      
    '''
    def drop_ST_PT(self, df):

        result = df.copy()
        result = result.merge(self.ST_PT, on=['Stkcd', 'Trdmnt'])
        result.drop(index=result[(result.Markettype == 2) | (result.Markettype == 8)].index, inplace=True)#上证B、深证B
        result.drop(index=result[
            (result.Trdsta != 1) & (result.Trdsta != 4) & (result.Trdsta != 7) & (result.Trdsta != 10) & (
                    result.Trdsta != 13)].index, inplace=True)
        return result

    def get_data(self, table, field=None, fs_freq='q'):

        if self.mode == 'local':
            raw_data = pd.read_csv(self.local_data_path + table + '.csv')

            if table == 'TRD_Mnth': #个股月收益率
                raw_data.Trdmnt = raw_data.Trdmnt.apply(lambda x: x[0:4] + x[5:7])
                raw_data = self.drop_ST_PT(raw_data)
                data = raw_data.pivot(index='Trdmnt', columns='Stkcd', values=field)

            if table == 'TRD_Dalyr': #个股日收益率
                raw_data.Trddt = raw_data.Trddt.apply(lambda x: x[0:4] + x[5:7] + x[8:10])
                raw_data.drop(index=raw_data[(raw_data.Markettype == 2) | (raw_data.Markettype == 8)].index,
                              inplace=True)
                raw_data.drop(index=raw_data[
                    (raw_data.Trdsta != 1) & (raw_data.Trdsta != 4) & (raw_data.Trdsta != 7) & (
                            raw_data.Trdsta != 10) & (raw_data.Trdsta != 13)].index, inplace=True)
                data = raw_data.pivot(index='Trddt', columns='Stkcd', values=field)

            if table == 'TRD_Cndalym': #综合市场交易数据
                raw_data.Trddt = raw_data.Trddt.apply(lambda x: x[0:4] + x[5:7] + x[8:10])
                raw_data = raw_data[raw_data.Markettype == 53]
                data = raw_data[['Trddt', field]]

            if table == 'TRD_Nrrate': #无风险利率
                raw_data['Trddt'] = raw_data['Clsdt']
                raw_data.Trddt = raw_data.Trddt.apply(lambda x: x[0:4] + x[5:7] + x[8:10])
                data = raw_data[['Trddt', field]]
                data[field] = data[field].apply(lambda x: x / 100)

            if table == 'STK_MKT_THRFACDAY': #三因子收益率
                raw_data = raw_data[raw_data.MarkettypeID == 'P9714']
                raw_data['Trddt'] = raw_data['TradingDate']
                raw_data.Trddt = raw_data.Trddt.apply(lambda x: x[0:4] + x[5:7] + x[8:10])
                data = raw_data[['Trddt', 'RiskPremium1', 'SMB1', 'HML1']]

            if table == 'STK_MKT_FIVEFACDAY': #五因子收益率
                raw_data = raw_data[(raw_data.MarkettypeID == 'P9714') & (raw_data.Portfolios == 1)]
                raw_data['Trddt'] = raw_data['TradingDate']
                raw_data.Trddt = raw_data.Trddt.apply(lambda x: x[0:4] + x[5:7] + x[8:10])
                data = raw_data[['Trddt', 'RiskPremium1', 'SMB1', 'HML1','RMW1','CMA1']]

            if table == 'CG_Co' : #公司研究-治理结构-公司基本情况文件
                raw_data = raw_data[raw_data.Stktype != 'B']
                raw_data['ListedDate'] = raw_data.ListedDate.apply(lambda x: x[0:4] + x[5:7])
                data = raw_data[['Stkcd',field]]
                data.set_index('Stkcd', inplace=True)

            if table == 'CG_Ybasic' : #公司研究-治理结构-治理综合信息文件
                mon_ret = self.monthly_ret.copy()

                raw_data['Annodt'] = raw_data.Annodt.apply(lambda x: x[0:4] + x[5:7])
                data = raw_data.pivot(index='Annodt', columns='Stkcd', values=field)
                col_list = list(data.iloc[10].index)
                drop_list = []
                for i in col_list:
                    if i not in list(mon_ret.iloc[10].index):
                        drop_list.append(i)
                data = data.drop(columns =drop_list)
                data.fillna(method='ffill', inplace=True)


            if table == 'TRD_Co': #股票市场交易-基本数据-公司文件
                data = raw_data[['Stkcd', field]]
                data.set_index('Stkcd', inplace=True)

            if table == 'IAR_Rept' : #公司研究系列-年中季报公布日期-基本情况文件
                raw_data.Annodt = raw_data.Annodt.apply(lambda x: x[0:4] + x[5:7] + x[8:10])
                data = raw_data[['Stkcd',field]]
                #资产负债表|利润表|现金流量表(直接法)|现金流量表(间接法)
            if (table == 'FS_Combas') | (table == 'FS_Comins') | (table == 'FS_Comscfd') | (table == 'FS_Comscfi'):
                raw_data = pd.read_csv(self.local_data_path + table + '.csv')
                raw_data.drop(index=raw_data[raw_data.Typrep == 'B'].index, inplace=True)#去除母公司报表
                raw_data['type'] = raw_data.Accper.apply(lambda x: x[5:7] + x[8:10])#统计截止日期

                if fs_freq == 'y':#年度
                    raw_data.drop(index=raw_data[raw_data.type != '1231'].index, inplace=True)  #不存在type字段？？？？？
                    raw_data.Accper = raw_data.Accper.apply(lambda x: x[0:4] + x[5:7])
                    raw_data = raw_data.pivot(index='Accper', columns='Stkcd', values=field)

                if fs_freq == 'q':#季度
                    raw_data.drop(index=raw_data[
                        (raw_data.type != '1231') & (raw_data.type != '0930') & (raw_data.type != '0630') & (
                                raw_data.type != '0331')].index, inplace=True)
                    raw_data.Accper = raw_data.Accper.apply(lambda x: x[0:4] + x[5:7])
                    raw_data = raw_data.pivot(index='Accper', columns='Stkcd', values=field)

                    if table != 'FS_Combas':    #不是资产负债表，当季度数据设定为当季与上一季的差值（三月除外）
                        dif_lag_data = raw_data.copy()
                        for i in range(raw_data.shape[0]):
                            if (raw_data.index[i][4:] == '09') or (raw_data.index[i][4:] == '12'):
                                raw_data.iloc[i] = dif_lag_data.iloc[i] - dif_lag_data.iloc[i - 1]
                            if (raw_data.index[i][4:] == '06') and i>0:
                                if raw_data.index[i-1][4:] == '03':
                                    raw_data.iloc[i] = dif_lag_data.iloc[i] - dif_lag_data.iloc[i - 1]

                universe = self.monthly_ret.copy()
                data = universe.copy()
                data.iloc[:, :] = np.nan
                lag_data = raw_data.reset_index()
                lag_d = []
                num = lag_data.shape[0]


                for i in range(num):        #月份修改
                    if lag_data.loc[i]['Accper'][4:] == '03':
                        lag_d.append(lag_data.loc[i]['Accper'][0:4] + '04')
                    if lag_data.loc[i]['Accper'][4:] == '06':
                        lag_d.append(lag_data.loc[i]['Accper'][0:4] + '08')
                    if lag_data.loc[i]['Accper'][4:] == '09':
                        lag_d.append(lag_data.loc[i]['Accper'][0:4] + '10')
                    if lag_data.loc[i]['Accper'][4:] == '12':
                        newyear = str(int(lag_data.loc[i]['Accper'][0:4]) + 1)
                        lag_d.append(newyear + '04')

                lag_data['Accper'] = pd.DataFrame(lag_d)
                lag_data.fillna(method='ffill', inplace=True)
                lag_data.drop_duplicates(subset=['Accper'], keep='last', inplace=True)
                lag_data.set_index(['Accper'], inplace=True)

                data.loc[:, :] = lag_data.loc[:, :]
                data.fillna(method='ffill', inplace=True)
                universe = np.isnan(universe)
                data[universe] = np.nan

        return data

    def calc_momtest(self, start=2, end=12):
        ret = self.monthly_ret.copy()
        result = 1
        for i in range(start, end + 1):
            temp_lag = ret.shift(i)
            result = result * (1 + temp_lag)
        result = result - 1
        return result

    ###################################################################################
    # factor calculation
    # B.1 Trading Related Anomalies

    # B.1 Liquidity (13)
    # B.1.1.1 Firm Size(size)
    def calc_size(self):
        lncap = self.get_data('TRD_Mnth', 'Msmvosd')#月个股流通市值数据
        lncap = np.log(lncap)
        return lncap

    def calc_size3(self):
        midcap = self.get_data('TRD_Mnth', 'Msmvosd')
        midcap = np.power(np.log(midcap), 3)#三次方
        return midcap

    # B.1.1.2 Share Turnover (turn1, turn6, and turn12)

    # month
    def calc_turnm(self):
        volume = self.get_data('TRD_Mnth', 'Mnshrtrd')
        mv = self.get_data('TRD_Mnth', 'Msmvosd')
        price = self.get_data('TRD_Mnth', 'Mclsprc')
        number = mv/price
        turnover = np.log(volume / number)
        return turnover

    # quarter
    def calc_turnq(self):
        volume = self.get_data('TRD_Mnth', 'Mnshrtrd')
        mv = self.get_data('TRD_Mnth', 'Msmvosd')
        price = self.get_data('TRD_Mnth', 'Mclsprc')
        number = mv / price
        turnover = np.log(volume / number)
        result = np.log(turnover.rolling(3).mean())
        return result

    # annual
    def calc_turna(self):
        volume = self.get_data('TRD_Mnth', 'Mnshrtrd')
        mv = self.get_data('TRD_Mnth', 'Msmvosd')
        price = self.get_data('TRD_Mnth', 'Mclsprc')
        number = mv / price
        turnover = np.log(volume / number)
        result = np.log(turnover.rolling(12, min_periods=4).mean())
        return result

    # B.1.1.3 Variation of Share Turnover (vturn1, vturn6, and vturn12)
    def calc_vturn(self, trading_day_num=120, min_day_num=40):

        volume = self.get_data('TRD_Dalyr', 'Dnshrtrd')
        mv = self.get_data('TRD_Dalyr', 'Dsmvosd')
        price = self.get_data('TRD_Dalyr', 'Clsprc')
        number = mv/price
        turnover = volume / number
        result = turnover.rolling(trading_day_num, min_periods=min_day_num).std()

        result['Trdmnt'] = list(result.index)
        result.Trdmnt = result.Trdmnt.apply(lambda x: x[0:6])
        result.drop_duplicates(subset=['Trdmnt'], keep='last', inplace=True)
        result.set_index('Trdmnt', inplace=True)

        return result

    #B.1.1.4 Coefficient of Variation of Share Turnover (cvturn1, cvturn6, and cvturn12)
    # the ratio of the standard deviation to the mean of daily share turnover
    def calc_cvturn(self, trading_day_num=120, min_day_num=40):
        volume = self.get_data('TRD_Dalyr', 'Dnshrtrd')
        mv = self.get_data('TRD_Dalyr', 'Dsmvosd')
        price = self.get_data('TRD_Dalyr', 'Clsprc')
        number = mv / price
        turnover = volume / number
        sd = turnover.rolling(trading_day_num, min_periods=min_day_num).std()
        mean = turnover.rolling(trading_day_num, min_periods=min_day_num).mean()
        result = sd/mean

        result['Trdmnt'] = list(result.index)
        result.Trdmnt = result.Trdmnt.apply(lambda x: x[0:6])
        result.drop_duplicates(subset=['Trdmnt'], keep='last', inplace=True)
        result.set_index('Trdmnt', inplace=True)

        return result

    # B.1.1.5 One-month Abnormal Turnover (abturn)
    # the ratio of average daily turnover in month t to its average daily turnover in prior one year over t-11 to t.
    def calc_abturn(self, trading_month_num=21, trading_year_num=250, min_month_num=7,min_year_num=85 ):

        volume = self.get_data('TRD_Dalyr', 'Dnshrtrd')
        mv = self.get_data('TRD_Dalyr', 'Dsmvosd')
        price = self.get_data('TRD_Dalyr', 'Clsprc')
        number = mv / price
        turnover = volume / number
        month_t = turnover.rolling(trading_month_num, min_periods=min_month_num).mean()
        year_t = turnover.rolling(trading_year_num, min_periods=min_year_num).mean()

        result = month_t / year_t

        result['Trdmnt'] = list(result.index)
        result.Trdmnt = result.Trdmnt.apply(lambda x: x[0:6])
        result.drop_duplicates(subset=['Trdmnt'], keep='last', inplace=True)
        result.set_index('Trdmnt', inplace=True)

        return result

    # B.1.1.6 Dollar Trading Volume (dtv1, dtv6, and dtv12)
    # month
    def calc_dtvm(self):
        volume = self.get_data('TRD_Mnth', 'Mnvaltrd')
        result = np.log(volume)
        return result

    # quarter
    def calc_dtvq(self):
        volume = self.get_data('TRD_Mnth', 'Mnvaltrd')
        result = np.log(volume.rolling(3).mean())
        return result
    # annual
    def calc_dtva(self):
        volume = self.get_data('TRD_Mnth', 'Mnvaltrd')
        result = np.log(volume.rolling(12, min_periods=4).mean())
        return result

    # B.1.1.7 Variation of Dollar Trading Volume (vdtv1, vdtv6, and vdtv12)
    def calc_vdtv(self, trading_day_num=120, min_day_num=40):

        volume = self.get_data('TRD_Dalyr', 'Dnvaltrd')
        result = volume.rolling(trading_day_num, min_periods=min_day_num).std()

        result['Trdmnt'] = list(result.index)
        result.Trdmnt = result.Trdmnt.apply(lambda x: x[0:6])
        result.drop_duplicates(subset=['Trdmnt'], keep='last', inplace=True)
        result.set_index('Trdmnt', inplace=True)

        return result

    # B.1.1.8 Coefficient of Variation of Dollar Trading Volume (cvd1, cvd6, and cvd12)
    def calc_cvd(self, trading_day_num=120, min_day_num=40):
        volume = self.get_data('TRD_Dalyr', 'Dnvaltrd')
        sd = volume.rolling(trading_day_num, min_periods=min_day_num).std()
        mean = volume.rolling(trading_day_num, min_periods=min_day_num).mean()
        result = sd/mean

        result['Trdmnt'] = list(result.index)
        result.Trdmnt = result.Trdmnt.apply(lambda x: x[0:6])
        result.drop_duplicates(subset=['Trdmnt'], keep='last', inplace=True)
        result.set_index('Trdmnt', inplace=True)

        return result

    # B.1.1.9 Amihud Illiquidity (Absolute Return-to-Volume) (Ami1, Ami6, and Ami12)
    def calc_Ami(self):
        volume = self.get_data('TRD_Mnth', 'Mnvaltrd')
        ret = abs(self.monthly_ret.copy())
        Ami = ret / volume
        return Ami


    # B.1.2 Risk (13)

    # B.1.2.1 Idiosyncratic Volatility (idv)
    # idiosyncratic volatility per the CAPM, idvc
    def calc_idvc(self, trading_day_num=120, min_day_num=40):

        daily_ret = self.get_data('TRD_Dalyr', 'Dretwd')#日回报率
        market_ret = self.get_data('TRD_Cndalym', 'Cdretwdos')#综合日市场回报率
        result = daily_ret.copy()
        result.iloc[:, :] = np.nan

        rf = self.get_data('TRD_Nrrate', 'Nrrdaydt')#日度无风险利率
        market_and_rf = market_ret.merge(rf, on=['Trddt'])

        rf = np.array(market_and_rf.iloc[:, -1])
        x = np.array(market_and_rf.iloc[:, -2]) - rf
        x = sm.add_constant(x)

        for j in range(result.shape[1]):
            # print(j)
            y = np.array(daily_ret.iloc[:, j]) - rf
            model = RollingOLS(y, x, window=trading_day_num, min_nobs=min_day_num).fit()#个股对市场的滚动回归
            result.iloc[:, j] = np.sqrt(model.mse_resid * model.df_resid / (model.df_model + 1 + model.df_resid))
        result['Trdmnt'] = list(result.index)
        result.Trdmnt = result.Trdmnt.apply(lambda x: x[0:6])
        result.drop_duplicates(subset=['Trdmnt'], keep='last', inplace=True)
        result.set_index('Trdmnt', inplace=True)

        return result

    # B.1.2.2 Idiosyncratic Volatility per the CH3 Factor Model (idvff)
    def calc_idvff(self, trading_day_num=120, min_day_num=40):

        daily_ret = self.get_data('TRD_Dalyr', 'Dretwd')
        ff3 = self.get_data('STK_MKT_THRFACDAY')
        result = daily_ret.copy()
        result.iloc[:, :] = np.nan

        rf = self.get_data('TRD_Nrrate', 'Nrrdaydt')

        ff3_and_rf = ff3.merge(rf, on=['Trddt'])

        # ['Trddt', 'RiskPremium1', 'SMB1', 'HML1', 'rf']
        rf = np.array(ff3_and_rf.iloc[:, -1])

        x = np.array(ff3_and_rf.iloc[:, -4:-1])

        x = sm.add_constant(x)

        for j in range(result.shape[1]):
            # print(j)
            y = np.array(daily_ret.iloc[:, j]) - rf
            model = RollingOLS(y, x, window=trading_day_num, min_nobs=min_day_num).fit()
            result.iloc[:, j] = np.sqrt(model.mse_resid * model.df_resid / (model.df_model + 1 + model.df_resid))
        result['Trdmnt'] = list(result.index)
        result.Trdmnt = result.Trdmnt.apply(lambda x: x[0:6])
        result.drop_duplicates(subset=['Trdmnt'], keep='last', inplace=True)
        result.set_index('Trdmnt', inplace=True)

        return result

    # B.1.2.3 Idiosyncratic Volatility per the q-factor Model (idvq)
    def calc_idvq(self, trading_day_num=120, min_day_num=40):

        daily_ret = self.get_data('TRD_Dalyr', 'Dretwd')
        q_factor = self.get_data('STK_MKT_FIVEFACDAY')
        result = daily_ret.copy()
        result.iloc[:, :] = np.nan

        q_factor = daily_ret.merge(q_factor, how = 'outer',on=['Trddt'])
        q_factor = q_factor[['Trddt', 'RiskPremium1', 'SMB1','RMW1','CMA1']]

        rf = self.get_data('TRD_Nrrate', 'Nrrdaydt')

        q_factor_and_rf = q_factor.merge(rf, on=['Trddt'])

        # ['Trddt', 'RiskPremium1', 'SMB1', 'RMW1','CMA1', 'rf']
        rf = np.array(q_factor_and_rf.iloc[:, -1])

        x = np.array(q_factor_and_rf.iloc[:, -5:-1])
        x = sm.add_constant(x)

        for j in range(result.shape[1]):
            # print(j)
            y = np.array(daily_ret.iloc[:, j]) - rf
            model = RollingOLS(y, x, window=trading_day_num, min_nobs=min_day_num).fit()
            result.iloc[:, j] = np.sqrt(model.mse_resid * model.df_resid / (model.df_model + 1 + model.df_resid))
        result['Trdmnt'] = list(result.index)
        result.Trdmnt = result.Trdmnt.apply(lambda x: x[0:6])
        result.drop_duplicates(subset=['Trdmnt'], keep='last', inplace=True)
        result.set_index('Trdmnt', inplace=True)

        return result


    # B.1.2.5 Total Volatility (tv)
    def calc_tv(self, trading_day_num=120, min_day_num=40):
        daily_ret = self.get_data('TRD_Dalyr', 'Dretwd')
        result = daily_ret.rolling(trading_day_num, min_periods=min_day_num).std()
        result['Trdmnt'] = list(result.index)
        result.Trdmnt = result.Trdmnt.apply(lambda x: x[0:6])
        result.drop_duplicates(subset=['Trdmnt'], keep='last', inplace=True)
        result.set_index('Trdmnt', inplace=True)
        return result

    # B.1.2.6 Idiosyncratic Skewness per the CH3 Factor Model (idsff)

    def calc_idsff(self, trading_day_num=120, min_day_num=40):

        daily_ret = self.get_data('TRD_Dalyr', 'Dretwd')
        ff3 = self.get_data('STK_MKT_THRFACDAY')
        rf = self.get_data('TRD_Nrrate', 'Nrrdaydt')
        ff3_and_rf = ff3.merge(rf, on=['Trddt'])
        rf = np.array(ff3_and_rf.iloc[:, -1])
        x = np.array(ff3_and_rf.iloc[:, -4:-1])
        x = sm.add_constant(x)

        sk = daily_ret.copy()
        sk.iloc[:, :] = np.nan
        sk['Trdmnt'] = list(sk.index)
        sk.Trdmnt = sk.Trdmnt.apply(lambda x: x[0:6])
        sk.drop_duplicates(subset=['Trdmnt'], keep='last', inplace=True)

        for j in range(sk.shape[1] - 1):
            # print(j)
            y = np.array(daily_ret.iloc[:, j]) - rf
            model = RollingOLS(y, x, window=trading_day_num, min_nobs=min_day_num).fit()

            iv = np.sqrt(model.mse_resid * model.df_resid / (model.df_model + 1 + model.df_resid))#mse_resid残差均方误差,df_resid残差自由度,df_model模型自由度
            num = model.df_model + 1 + model.df_resid

            for m in range(sk.shape[0]):
                pos = list(daily_ret.index).index(sk.index[m])#确定每个月份最后一天在daily_ret中的位置
                temp_x = x[pos - trading_day_num: pos, :]#将滚动窗口扩展到“负位置”/当前月末之前的窗口
                temp_y = y[pos - trading_day_num: pos]
                temp_param = model.params[pos]
                valid_list = ~np.isnan(temp_y)
                temp_y = temp_y[valid_list]
                temp_x = temp_x[valid_list]

                temp_pre = np.dot(temp_x, temp_param)#预测
                temp_res = temp_y - temp_pre#预测误差

                temp_iv = iv[pos]
                temp_num = num[pos]
                temp_is = np.sum(np.power(temp_res, 3)) / (temp_num * np.power(temp_iv, 3))#求特异质偏度
                sk.iloc[m, j] = temp_is

        sk.set_index('Trdmnt', inplace=True)
        return sk

    # B.1.2.7 Idiosyncratic Skewness per the q-Factor Model (idsq)
    def calc_idsq(self, trading_day_num=120, min_day_num=40):

        daily_ret = self.get_data('TRD_Dalyr', 'Dretwd')
        q_factor = self.get_data('STK_MKT_FIVEFACDAY')
        result = daily_ret.copy()
        result.iloc[:, :] = np.nan

        q_factor = daily_ret.merge(q_factor, how='outer', on=['Trddt'])
        q_factor = q_factor[['Trddt', 'RiskPremium1', 'SMB1', 'RMW1', 'CMA1']]

        rf = self.get_data('TRD_Nrrate', 'Nrrdaydt')
        q_factor_and_rf = q_factor.merge(rf, on=['Trddt'])
        # ['Trddt', 'RiskPremium1', 'SMB1', 'RMW1','CMA1', 'rf']
        rf = np.array(q_factor_and_rf.iloc[:, -1])
        x = np.array(q_factor_and_rf.iloc[:, -5:-1])
        x = sm.add_constant(x)

        sk = daily_ret.copy()
        sk.iloc[:, :] = np.nan
        sk['Trdmnt'] = list(sk.index)
        sk.Trdmnt = sk.Trdmnt.apply(lambda x: x[0:6])
        sk.drop_duplicates(subset=['Trdmnt'], keep='last', inplace=True)

        for j in range(sk.shape[1] - 1):
            # print(j)
            y = np.array(daily_ret.iloc[:, j]) - rf
            model = RollingOLS(y, x, window=trading_day_num, min_nobs=min_day_num).fit()

            iv = np.sqrt(model.mse_resid * model.df_resid / (model.df_model + 1 + model.df_resid))
            num = model.df_model + 1 + model.df_resid

            for m in range(sk.shape[0]):
                pos = list(daily_ret.index).index(sk.index[m])
                temp_x = x[pos - trading_day_num: pos, :]
                temp_y = y[pos - trading_day_num: pos]
                temp_param = model.params[pos]
                valid_list = ~np.isnan(temp_y)
                temp_y = temp_y[valid_list]
                temp_x = temp_x[valid_list]

                temp_pre = np.dot(temp_x, temp_param)
                temp_res = temp_y - temp_pre

                temp_iv = iv[pos]
                temp_num = num[pos]
                temp_is = np.sum(np.power(temp_res, 3)) / (temp_num * np.power(temp_iv, 3))
                sk.iloc[m, j] = temp_is

        sk.set_index('Trdmnt', inplace=True)
        return sk

    # B.1.2.8 Idiosyncratic Skewness per the CAPM (idsc)
    def calc_idsc(self, trading_day_num=120, min_day_num=40):

        daily_ret = self.get_data('TRD_Dalyr', 'Dretwd')
        market_ret = self.get_data('TRD_Cndalym', 'Cdretwdos')
        rf = self.get_data('TRD_Nrrate', 'Nrrdaydt')
        market_and_rf = market_ret.merge(rf, on=['Trddt'])
        rf = np.array(market_and_rf.iloc[:, -1])
        x = np.array(market_and_rf.iloc[:, -2]) - rf
        x = sm.add_constant(x)

        sk = daily_ret.copy()
        sk.iloc[:, :] = np.nan
        sk['Trdmnt'] = list(sk.index)
        sk.Trdmnt = sk.Trdmnt.apply(lambda x: x[0:6])
        sk.drop_duplicates(subset=['Trdmnt'], keep='last', inplace=True)

        for j in range(sk.shape[1] - 1):
            # print(j)
            y = np.array(daily_ret.iloc[:, j]) - rf
            model = RollingOLS(y, x, window=trading_day_num, min_nobs=min_day_num).fit()

            iv = np.sqrt(model.mse_resid * model.df_resid / (model.df_model + 1 + model.df_resid))
            num = model.df_model + 1 + model.df_resid

            for m in range(sk.shape[0]):
                pos = list(daily_ret.index).index(sk.index[m])
                temp_x = x[pos - trading_day_num: pos, :]
                temp_y = y[pos - trading_day_num: pos]
                temp_param = model.params[pos]
                valid_list = ~np.isnan(temp_y)
                temp_y = temp_y[valid_list]
                temp_x = temp_x[valid_list]

                temp_pre = np.dot(temp_x, temp_param)
                temp_res = temp_y - temp_pre

                temp_iv = iv[pos]
                temp_num = num[pos]
                temp_is = np.sum(np.power(temp_res, 3)) / (temp_num * np.power(temp_iv, 3))
                sk.iloc[m, j] = temp_is

        sk.set_index('Trdmnt', inplace=True)
        return sk

    # B.1.2.9 Total Skewness (ts)
    def calc_ts(self, trading_day_num=120, min_day_num=40):
        daily_ret = self.get_data('TRD_Dalyr', 'Dretwd')
        result = daily_ret.rolling(trading_day_num, min_periods=min_day_num).skew()
        result['Trdmnt'] = list(result.index)
        result.Trdmnt = result.Trdmnt.apply(lambda x: x[0:6])
        result.drop_duplicates(subset=['Trdmnt'], keep='last', inplace=True)
        result.set_index('Trdmnt', inplace=True)
        return result

    # B.1.2.10 Co-skewness (cs)
    def calc_cs(self, trading_day_num=120, min_day_num=40):

        daily_ret = self.get_data('TRD_Dalyr', 'Dretwd')
        market_ret = self.get_data('TRD_Cndalym', 'Cdretwdos')
        rf = self.get_data('TRD_Nrrate', 'Nrrdaydt')
        market_and_rf = market_ret.merge(rf, on=['Trddt'])
        rf = np.array(market_and_rf.iloc[:, -1])
        x = np.array(market_and_rf.iloc[:, -2]) - rf
        x = sm.add_constant(x)

        cs = daily_ret.copy()
        cs.iloc[:, :] = np.nan
        cs['Trdmnt'] = list(cs.index)
        cs.Trdmnt = cs.Trdmnt.apply(lambda x: x[0:6])
        cs.drop_duplicates(subset=['Trdmnt'], keep='last', inplace=True)

        for j in range(cs.shape[1] - 1):
            # print(j)
            y = np.array(daily_ret.iloc[:, j]) - rf
            model = RollingOLS(y, x, window=trading_day_num, min_nobs=min_day_num).fit()

            for m in range(cs.shape[0]):
                pos = list(daily_ret.index).index(cs.index[m])
                temp_x = x[pos - trading_day_num: pos, :]
                temp_y = y[pos - trading_day_num: pos]
                temp_param = model.params[pos]
                valid_list = ~np.isnan(temp_y)
                temp_y = temp_y[valid_list]
                temp_x = temp_x[valid_list]
                temp_pre = np.dot(temp_x, temp_param)
                temp_res = temp_y - temp_pre
                temp_mkt = temp_x[:, -1] - np.nanmean(temp_x[:, -1])

                temp_cs = np.nanmean(temp_res * temp_mkt * temp_mkt) / (
                        np.sqrt(np.nanmean(temp_res * temp_res)) * np.nanmean(temp_mkt * temp_mkt))
                cs.iloc[m, j] = temp_cs

        cs.set_index('Trdmnt', inplace=True)
        return cs

    # B.1.2.12 Market Beta Using Daily Returns (beta)
    def calc_beta(self, trading_day_num=120, min_day_num=40):

        daily_ret = self.get_data('TRD_Dalyr', 'Dretwd')
        market_ret = self.get_data('TRD_Cndalym', 'Cdretwdos')
        result = daily_ret.copy()
        result.iloc[:, :] = np.nan

        rf = self.get_data('TRD_Nrrate', 'Nrrdaydt')
        market_and_rf = market_ret.merge(rf, on=['Trddt'])

        rf = np.array(market_and_rf.iloc[:, -1])
        x = np.array(market_and_rf.iloc[:, -2]) - rf
        x = sm.add_constant(x)

        for j in range(result.shape[1]):
            # print(j)
            y = np.array(daily_ret.iloc[:, j]) - rf
            model = RollingOLS(y, x, window=trading_day_num, min_nobs=min_day_num).fit()
            result.iloc[:, j] = model.params[:, -1]

        result['Trdmnt'] = list(result.index)
        result.Trdmnt = result.Trdmnt.apply(lambda x: x[0:6])
        result.drop_duplicates(subset=['Trdmnt'], keep='last', inplace=True)
        result.set_index('Trdmnt', inplace=True)

        return result

    # for @xinhe chinese 4 factor
    def rolling_ols_four_chbeta(self,trading_day_num=120, min_day_num=40):
        ch4 = pd.read_csv(self.local_data_path+'CH_4_fac_daily_update_20211231.csv')
        ch4.rename(columns={'date':'Trddt'},inplace = True)
        ch4['Trddt'] = ch4['Trddt'].astype(str)
        daily_ret = self.get_data('TRD_Dalyr', 'Dretwd')
        ch4 = pd.merge(pd.DataFrame(daily_ret.index).astype(str),ch4,how = 'outer')
        rf = self.get_data('TRD_Nrrate', 'Nrrdaydt')
        daily_and_rf = daily_ret.merge(rf, on=['Trddt'])
        rf = np.array(daily_and_rf.iloc[:, -1])

        x = np.array(ch4.iloc[:,2:]/100)#四因子，跳过第一列的日期和第二列的无风险利率
        x=  sm.add_constant(x)

        result_mktrf = daily_ret.copy()
        result_mktrf.iloc[:, :] = np.nan

        result_vmg = daily_ret.copy()
        result_vmg.iloc[:, :] = np.nan

        result_smb = daily_ret.copy()
        result_smb.iloc[:, :] = np.nan

        result_pmo = daily_ret.copy()
        result_pmo.iloc[:, :] = np.nan

        #print("start rolling")

        for j in range(result_mktrf.shape[1]):
            #print('stock'+str(j))
            y = np.array(daily_ret.iloc[:, j]) - rf
            try:
                model = RollingOLS(y, x, window=trading_day_num, min_nobs=min_day_num).fit()
                result_mktrf.iloc[:, j] = model.params[:, 1]
                result_vmg.iloc[:, j] = model.params[:, 2]
                result_smb.iloc[:, j] = model.params[:, 3]
                result_pmo.iloc[:, j] = model.params[:, 4]
            except:
                pass
        #print("over rolling")
        for result in (result_mktrf,result_vmg,result_smb,result_pmo):
            result['Trdmnt'] = list(result.index)
            result.Trdmnt = result.Trdmnt.apply(lambda x: x[0:6])
            result.drop_duplicates(subset=['Trdmnt'], keep='last', inplace=True)#只取每月最后一天
            result.set_index('Trdmnt', inplace=True)

        self.mktrf_beta = result_mktrf
        self.vmg_beta = result_vmg
        self.smb_beta = result_smb
        self.pmo_beta = result_pmo

    def calc_mkt_beta(self):
        try:
            return self.mktrf_beta
        except:
            self.rolling_ols_four_chbeta()
            return self.mktrf_beta

    def calc_vmg_beta(self):
        try:
            return self.vmg_beta
        except:
            self.rolling_ols_four_chbeta()
            return self.vmg_beta

    def calc_smb_beta(self):
        try:
            return self.smb_beta
        except:
            self.rolling_ols_four_chbeta()
            return self.smb_beta

    def calc_pmo_beta(self):
        try:
            return self.pmo_beta
        except:
            self.rolling_ols_four_chbeta()
            return self.pmo_beta

    # B.1.2.13 Downside Beta (dbeta)
    def calc_dbeta(self, trading_day_num=240, min_day_num=80):

        daily_ret = self.get_data('TRD_Dalyr', 'Dretwd')
        market_ret = self.get_data('TRD_Cndalym', 'Cdretwdos')
        rf = self.get_data('TRD_Nrrate', 'Nrrdaydt')
        market_and_rf = market_ret.merge(rf, on=['Trddt'])
        rf = np.array(market_and_rf.iloc[:, -1])
        mkt = np.array(market_and_rf.iloc[:, -2]) - rf

        dbeta = daily_ret.copy()
        dbeta.iloc[:, :] = np.nan
        dbeta['Trdmnt'] = list(dbeta.index)
        dbeta.Trdmnt = dbeta.Trdmnt.apply(lambda x: x[0:6])
        dbeta.drop_duplicates(subset=['Trdmnt'], keep='last', inplace=True)

        for j in range(dbeta.shape[1] - 1):
            # print(j)
            y = np.array(daily_ret.iloc[:, j]) - rf

            for m in range(dbeta.shape[0]):
                pos = list(daily_ret.index).index(dbeta.index[m])
                temp_mkt = mkt[pos - trading_day_num: pos]
                temp_y = y[pos - trading_day_num: pos]

                valid_list = ~np.isnan(temp_y)
                temp_y = temp_y[valid_list]
                temp_mkt = temp_mkt[valid_list]

                down_list = temp_mkt < np.nanmean(temp_mkt)
                temp_y = temp_y[down_list]
                temp_mkt = temp_mkt[down_list]

                if len(temp_y) >= min_day_num:
                    dbeta.iloc[m, j] = np.cov(temp_y, temp_mkt)[0][1] / np.var(temp_mkt)
                else:
                    dbeta.iloc[m, j] = np.nan

        dbeta.set_index('Trdmnt', inplace=True)
        return dbeta

    # B.1.2.14 The Frazzini-Pedersen Beta (betaFP)
    def calc_betafp(self, win=120, minn=40):

        daily_ret = self.get_data('TRD_Dalyr', 'Dretwd')
        market_ret = self.get_data('TRD_Cndalym', 'Cdretwdos')
        rf = self.get_data('TRD_Nrrate', 'Nrrdaydt')
        market_and_rf = market_ret.merge(rf, on=['Trddt'])
        rf = np.array(market_and_rf.iloc[:, -1])
        mkt = np.array(market_and_rf.iloc[:, -2]) - rf

        daily_excess = daily_ret.apply(lambda x: x - rf)
        daily_logexcess = np.log(1 + daily_excess)
        daily_sigma = daily_logexcess.rolling(win, min_periods=minn).std()
        mkt_sigma = np.array(pd.DataFrame(mkt).rolling(win, min_periods=minn).std())
        r3d = daily_logexcess.rolling(3).sum()

        betafp = daily_ret.copy()
        betafp.iloc[:, :] = np.nan
        betafp['Trdmnt'] = list(betafp.index)
        betafp.Trdmnt = betafp.Trdmnt.apply(lambda x: x[0:6])
        betafp.drop_duplicates(subset=['Trdmnt'], keep='last', inplace=True)

        for j in range(betafp.shape[1] - 1):
            # print(j)

            for m in range(betafp.shape[0]):
                pos = list(daily_ret.index).index(betafp.index[m])#某个月的日收益index对应位置的列表
                temp_daily_sigma = daily_sigma.iloc[pos, j]#提取对应月的每天波动率
                temp_mkt_sigma = mkt_sigma[pos][0]#市场波动率

                temp_mkt = mkt[pos - win: pos]
                temp_r3d = np.array(r3d.iloc[pos - win: pos, j])
                valid_list = ~np.isnan(temp_r3d)
                temp_r3d = temp_r3d[valid_list]
                temp_mkt = temp_mkt[valid_list]

                if len(temp_r3d) >= minn:
                    betafp.iloc[m, j] = np.corrcoef(temp_r3d, temp_mkt)[0][1] * temp_daily_sigma / temp_mkt_sigma
                else:
                    betafp.iloc[m, j] = np.nan

        betafp.set_index('Trdmnt', inplace=True)
        return betafp

    # B.1.2.15 The Dimson Beta (betaDM)
    def calc_betadm(self, trading_day_num=120, min_day_num=40):

        daily_ret = self.get_data('TRD_Dalyr', 'Dretwd')
        market_ret = self.get_data('TRD_Cndalym', 'Cdretwdos')
        result = daily_ret.copy()
        result.iloc[:, :] = np.nan

        rf = self.get_data('TRD_Nrrate', 'Nrrdaydt')
        market_and_rf = market_ret.merge(rf, on=['Trddt'])
        rf = np.array(market_and_rf.iloc[:, -1])
        market_and_rf.iloc[:, -2] = market_and_rf.iloc[:, -2] - rf

        x1 = np.array(market_and_rf.iloc[:, -2])
        x2 = np.array(market_and_rf.shift(-1).iloc[:, -2])#整体上移
        x3 = np.array(market_and_rf.shift(1).iloc[:, -2])#整体下移
        x = np.array([x1, x2, x3]).T
        x = sm.add_constant(x)

        for j in range(result.shape[1]):
            # print(j)
            y = np.array(daily_ret.iloc[:, j]) - rf
            if np.sum(~np.isnan(y)) >= 10:
                model = RollingOLS(y, x, window=trading_day_num, min_nobs=min_day_num).fit()
                result.iloc[:, j] = model.params[:, -1] + model.params[:, -2] + model.params[:, -3]

        result['Trdmnt'] = list(result.index)
        result.Trdmnt = result.Trdmnt.apply(lambda x: x[0:6])
        result.drop_duplicates(subset=['Trdmnt'], keep='last', inplace=True)
        result.set_index('Trdmnt', inplace=True)

        return result

    # B.1.2.16 Tail Risk (tail)
    # !!!Not tested yet
    def calc_tail(self,trading_month_num = 12,min_month_num = 6):
        daily_ret = self.get_data('TRD_Dalyr', 'Dretwd')
        market_ret = self.get_data('TRD_Cndalym', 'Cdretwdos')
        x = daily_ret.copy()
        x.iloc[:, :] = np.nan
        for i in range(daily_ret.shape[1]):
            for j in range(21,daily_ret.shape[0]):
                percent = np.percentile(daily_ret.iloc[j-21:j,i],5)#第5个百分位数
                k = 0
                lambd = 0
                for m in range(21):
                    if daily_ret.iloc[j-m,i] < percent:
                        lambd+=np.log(daily_ret.iloc[j-m,i])#每21天滚动窗口中小于第5个百分位数的日收益率之和
                        k+=1
                if k==0:
                    lambd = 0
                else:
                    lambd = lambd / k
                x[j,i] = lambd
        x['Trdmnt'] = list(x.index)
        x.Trdmnt = x.Trdmnt.apply(lambda x: x[0:6])
        x.drop_duplicates(subset=['Trdmnt'], keep='last', inplace=True)
        x.set_index('Trdmnt', inplace=True)#每月最后一天的21日内小于5%分位数收益率的平均

        rf = self.get_data('TRD_Nrrate', 'Nrrdaydt')
        market_and_rf = market_ret.merge(rf, on=['Trddt'])
        market_and_rf['Trdmnt'] = list(market_and_rf.index)
        market_and_rf.Trdmnt = market_and_rf.Trdmnt.apply(lambda x: x[0:6])
        market_and_rf.drop_duplicates(subset=['Trdmnt'], keep='last', inplace=True)#每月最后一天的市场收益率和无风险利率
        market_and_rf.set_index('Trdmnt', inplace=True)
        rf = np.array(market_and_rf.iloc[:, -1])

        ret = self.monthly_ret.copy()
        y = y.shift(1)
        y = sm.add_constant(y)
        result = self.monthly_ret.copy()
        result.iloc[:, :] = np.nan

        for j in range(result.shape[1]):
            y = np.array(ret.iloc[:, j]) - rf#第j只股票的月超额收益率
            model = RollingOLS(y, x, window=trading_month_num, min_nobs=min_month_num).fit()
            result[:, j] = model.params[:-1]#回归出尾部风险对月超额收益的系数


        return result

    # B.1.3 Past Returns (14)
    def calc_m1(self):
        result = self.monthly_ret.copy()
        return result

    # B.1.3.1 Prior k-month Momentum (m11, m9, m6, m3)
    def calc_m11(self, start=1, end=11):
        ret = self.monthly_ret.copy()
        result = 1
        for i in range(start, end + 1):
            temp_lag = ret.shift(i)
            result = result * (1 + temp_lag)
        result = result - 1
        return result

    def calc_m6(self, start=1, end=5):
        ret = self.monthly_ret.copy()
        result = 1
        for i in range(start, end + 1):
            temp_lag = ret.shift(i)
            result = result * (1 + temp_lag)
        result = result - 1
        return result

    def calc_m3(self, start=1, end=3):
        ret = self.monthly_ret.copy()
        result = 1
        for i in range(start, end + 1):
            temp_lag = ret.shift(i)
            result = result * (1 + temp_lag)
        result = result - 1
        return result

    # long term reversal
    def calc_m60(self,start = 12,end=59):
        ret = self.monthly_ret.copy()
        result = 1
        for i in range(start, end + 1):
            temp_lag = ret.shift(i)
            result = result * (1 + temp_lag)
        result = result - 1
        return result

    # B.1.3.2 Industry Momentum (inm)
    def calc_indmom(self, month_num=6, min_month_num=2):
        ind_table = self.get_data('TRD_Co','Nnindcd')#2012版证监会行业分类代码，股票代码是index

        stock_ind = self.monthly_ret.copy()
        for i in range(stock_ind.shape[1]):
            ind = list(ind_table.loc[stock_ind.columns[i]])[0]#股票行业
            stock_ind.iloc[:, i] = [ind] * stock_ind.shape[0]
        universe = self.monthly_ret.copy()
        universe = np.isnan(universe)
        stock_ind[universe] = np.nan

        ret = self.monthly_ret.copy()
        ret = np.log(ret + 1)
        rs = ret.rolling(month_num, min_periods=min_month_num).mean()#月对数收益率滚动平均

        mv = self.get_data('TRD_Mnth', 'Msmvosd')#月个股流通市值
        sqmv = np.sqrt(mv)

        ind_list = list(set(list(ind_table.iloc[:, 0])))#利用集合特性，获取唯一的行业代码

        indmom = self.monthly_ret.copy()
        indmom.iloc[:, :] = np.nan#制作回归系数表格
        rsi = indmom.copy()
        for i in range(stock_ind.shape[0]):
            # print(i)
            for j in ind_list:
                temp = stock_ind.iloc[i, :]
                pos = np.where(temp == j)[0]#第一个和行业列表对应行业相同的股票位置
                temp_sqmv = np.array(sqmv.iloc[i, pos])#第i月，第pos股票月流通市值
                temp_rs = np.array(rs.iloc[i, pos])
                temp_rsi = np.nansum(temp_rs * temp_sqmv) / np.nansum(temp_sqmv)#加和_of(月收益率*月流通市值^1/2)/加和_of月流通市值^1/2
                rsi.iloc[i, pos] = temp_rsi
        indmom = -(sqmv * rs - rsi)

        return indmom

    # B.1.3.3 Prior 24-month Momentum (m24)
    def calc_m24(self, start=12, end=35):
        ret = self.monthly_ret.copy()
        skip = np.isnan(ret)
        ret[skip] = 0

        result = 1
        for i in range(start, end + 1):
            temp_lag = ret.shift(i)
            result = result * (1 + temp_lag)
        result = result - 1
        result[skip] = np.nan

        return result

    # B.1.3.4 Momentum Change (mchg)
    def calc_mchg(self, s1=1, e1=6, s2=7, e2=12):
        ret = self.monthly_ret.copy()

        m1 = 1
        for i in range(s1, e1 + 1):
            temp_lag = ret.shift(i)
            m1 = m1 * (1 + temp_lag)
        m1 = m1 - 1

        m2 = 1
        for i in range(s2, e2 + 1):
            temp_lag = ret.shift(i)
            m2 = m2 * (1 + temp_lag)
        m2 = m2 - 1

        result = m1 - m2
        return result

    # B.1.3.7 k-month Residual Momentum (im11, im6)
    def calc_im12(self, im_day_num=240, trading_day_num=720, min_day_num=240):

        daily_ret = self.get_data('TRD_Dalyr', 'Dretwd')
        ff3 = self.get_data('STK_MKT_THRFACDAY')
        rf = self.get_data('TRD_Nrrate', 'Nrrdaydt')
        ff3_and_rf = ff3.merge(rf, on=['Trddt'])
        rf = np.array(ff3_and_rf.iloc[:, -1])
        x = np.array(ff3_and_rf.iloc[:, -4:-1])
        x = sm.add_constant(x)

        im = daily_ret.copy()
        im.iloc[:, :] = np.nan
        im['Trdmnt'] = list(im.index)
        im.Trdmnt = im.Trdmnt.apply(lambda x: x[0:6])
        im.drop_duplicates(subset=['Trdmnt'], keep='last', inplace=True)

        for j in range(im.shape[1] - 1):
            # print(j)
            y = np.array(daily_ret.iloc[:, j]) - rf
            model = RollingOLS(y, x, window=trading_day_num, min_nobs=min_day_num).fit()#超额收益、三因子收益

            for m in range(im.shape[0]):#横截面

                pos = list(daily_ret.index).index(im.index[m])#某月最后一天的日期在列表中的位置
                temp_x = x[pos - im_day_num: pos, :]#扩展滚动窗口
                temp_y = y[pos - im_day_num: pos]
                temp_param = model.params[pos]#每月最后一天的模型参数
                valid_list = ~np.isnan(temp_y)
                temp_y = temp_y[valid_list]
                temp_x = temp_x[valid_list]
                temp_pre = np.dot(temp_x, temp_param)#回归预测结果
                temp_res = temp_y - temp_pre#预测误差

                if np.nanstd(temp_res) != 0:
                    im.iloc[m, j] = np.nanmean(temp_res) / np.nanstd(temp_res)#预测误差的平均/预测误差的标准差
                else:
                    im.iloc[m, j] = np.nan

        im.set_index('Trdmnt', inplace=True)
        return im

    def calc_im6(self, im_day_num=120, trading_day_num=720, min_day_num=240):

        daily_ret = self.get_data('TRD_Dalyr', 'Dretwd')
        ff3 = self.get_data('STK_MKT_THRFACDAY')
        rf = self.get_data('TRD_Nrrate', 'Nrrdaydt')
        ff3_and_rf = ff3.merge(rf, on=['Trddt'])
        rf = np.array(ff3_and_rf.iloc[:, -1])
        x = np.array(ff3_and_rf.iloc[:, -4:-1])
        x = sm.add_constant(x)

        im = daily_ret.copy()
        im.iloc[:, :] = np.nan
        im['Trdmnt'] = list(im.index)
        im.Trdmnt = im.Trdmnt.apply(lambda x: x[0:6])
        im.drop_duplicates(subset=['Trdmnt'], keep='last', inplace=True)

        for j in range(im.shape[1] - 1):
            # print(j)
            y = np.array(daily_ret.iloc[:, j]) - rf
            model = RollingOLS(y, x, window=trading_day_num, min_nobs=min_day_num).fit()

            for m in range(im.shape[0]):

                pos = list(daily_ret.index).index(im.index[m])
                temp_x = x[pos - im_day_num: pos, :]
                temp_y = y[pos - im_day_num: pos]
                temp_param = model.params[pos]
                valid_list = ~np.isnan(temp_y)
                temp_y = temp_y[valid_list]
                temp_x = temp_x[valid_list]
                temp_pre = np.dot(temp_x, temp_param)
                temp_res = temp_y - temp_pre

                if np.nanstd(temp_res) != 0:
                    im.iloc[m, j] = np.nanmean(temp_res) / np.nanstd(temp_res)
                else:
                    im.iloc[m, j] = np.nan

        im.set_index('Trdmnt', inplace=True)
        return im

    # B.1.3.8 52-Week High (52w)
    def calc_52w(self):
        daily = self.get_data('TRD_Dalyr', 'Adjprcwd')#考虑现金红利再投资的收盘价的可比价格
        result = daily.copy()
        result.iloc[:, :] = np.nan
        result['Trdmnt'] = list(result.index)
        result.Trdmnt = result.Trdmnt.apply(lambda x: x[0:6])
        result.drop_duplicates(subset=['Trdmnt'], keep='last', inplace=True)

        for j in range(result.shape[1] - 1):
            for m in range(result.shape[0]):
                pos = list(daily.index).index(result.index[m])#获取一个月中最后一天的列表位置
                temp = np.array(daily.iloc[pos - 240:pos, j])
                if len(temp[~np.isnan(temp)]) > 0:            #存在有效数字
                    result.iloc[m, j] = temp[-1] / np.max(temp[~np.isnan(temp)])#每月最后一天收益率/滚动周期中的最大收益率
        result.set_index('Trdmnt', inplace=True)
        return result

    # B.1.3.9 Maximum Daily Return (mdr)
    def calc_mdr(self, mean_num=5, min_day_num=15, min_feb_num=5):

        daily_ret = self.get_data('TRD_Dalyr', 'Dretwd')
        daily_ret['Trdmnt'] = list(daily_ret.index)
        daily_ret.Trdmnt = daily_ret.Trdmnt.apply(lambda x: x[0:6])

        result = self.monthly_ret.copy()
        result.iloc[:, :] = np.nan

        for i in range(result.shape[0]):
            temp_data = daily_ret[daily_ret.Trdmnt == result.index[i]]#所有股票每月最后一天的日收益率面板
            for j in range(result.shape[1]):
                temp = np.array(temp_data.iloc[:, j])#每只股票
                not_nan_num = np.sum(~np.isnan(temp))

                if result.index[i][-2:] == '01' or result.index[i][-2:] == '02':#如果是每月的1号和2号
                    if not_nan_num >= min_feb_num:                  #非空值大于最小要求数量
                        result.iloc[i, j] = np.nanmean(temp[np.argsort(temp)[:not_nan_num]][-mean_num:])
                    else:
                        result.iloc[i, j] = np.nan
                else:
                    if not_nan_num >= min_day_num:                  #非空值大于最小要求数量
                        result.iloc[i, j] = np.nanmean(temp[np.argsort(temp)[:not_nan_num]][-mean_num:])#temp按大小排序后，前not_nan_num个非Nan值中倒数mean_num的值的平均
                    else:
                        result.iloc[i, j] = np.nan

        return result

    # B.1.3.10 Share Price (pr)
    def calc_pr(self):
        result = self.get_data('TRD_Dalyr', 'Adjprcwd')
        result['Trdmnt'] = list(result.index)
        result.Trdmnt = result.Trdmnt.apply(lambda x: x[0:6])
        result.drop_duplicates(subset=['Trdmnt'], keep='last', inplace=True)
        result.set_index('Trdmnt', inplace=True)
        return result

    # B.1.3.12 Cumulative Abnormal Returns around Earnings Announcement Dates (abr)
    # !!!Not tested yet
    def calc_abr(self):
        annouce_date = self.get_data('IAR_Rept','Annodt')#两列，左侧一列是股票代码，右侧是公告日期
        daily_ret = self.get_data('TRD_Dalyr', 'Dretwd')
        result = daily_ret.copy()
        result.iloc[:, :] = np.nan
        for j in range(daily_ret.shape[1]):
            for date in list(annouce_date[annouce_date['Stkcd'] == daily_ret.iloc[1, :].index[j]]['Annodt']):#遍历某只股票的公告日期。index[j]取的是columns名称
                if len(np.where(daily_ret.index == date)[0]) != 0:#公告日期是交易日
                    i = np.where(daily_ret.index == date)[0][0]#将array提取出[]列表再提取出数值(位置而不是日期)，确定公告日期在日收益率index的位置
                    if i==result.shape[0]-1:#如果是最后一天
                        result.iloc[i, j] = daily_ret.iloc[i - 2, j] + daily_ret.iloc[i - 1, j] + daily_ret.iloc[i, j]#前三天收益之和
                    else:
                        result.iloc[i, j] = daily_ret.iloc[i - 2, j] + daily_ret.iloc[i - 1, j] + daily_ret.iloc[i, j] + daily_ret.iloc[i + 1, j]#附近4天之和
                elif len(np.where(daily_ret.index == date[0:6] + str(int(date[6:9]) + 1))[0]) != 0:#公告日期后一天有交易
                    i = np.where(daily_ret.index == date[0:6] + str(int(date[6:9]) + 1)[0])[0]
                    if i==result.shape[0]-1:
                        result.iloc[i, j] = daily_ret.iloc[i - 2, j] + daily_ret.iloc[i - 1, j] + daily_ret.iloc[i, j]
                    else:
                        result.iloc[i, j] = daily_ret.iloc[i - 2, j] + daily_ret.iloc[i - 1, j] + daily_ret.iloc[i, j] + daily_ret.iloc[i + 1, j]
                elif len(np.where(daily_ret.index == date[0:6] + str(int(date[6:9]) - 1))[0]) != 0:#公告日期前一天有交易
                    i = np.where(daily_ret.index == date[0:6] + str(int(date[6:9]) - 1)[0])[0]
                    if i==result.shape[0]-1:
                        result.iloc[i, j] = daily_ret.iloc[i - 2, j] + daily_ret.iloc[i - 1, j] + daily_ret.iloc[i, j]
                    else:
                        result.iloc[i, j] = daily_ret.iloc[i - 2, j] + daily_ret.iloc[i - 1, j] + daily_ret.iloc[i, j] + daily_ret.iloc[i + 1, j]

        result.fillna(method='ffill', inplace=True)
        result['Trdmnt'] = list(result.index)
        result.Trdmnt = result.Trdmnt.apply(lambda x: x[0:6])
        result.drop_duplicates(subset=['Trdmnt'], keep='last', inplace=True)
        result.set_index('Trdmnt', inplace=True)

        return result

    # B.1.3.13 Seasonality (Ra1, Rn1, Ra25, Rn25)
    def calc_season(self, year_num=5, min_year_num=3):
        '''
        ret = self.monthly_ret.copy()
        result = 0
        for i in range(year_num):
            result = result + ret.shift(12*(i+1)-1)
        result = result/year_num
        return result
        '''
        ret = self.monthly_ret.copy()
        result = ret.copy()
        result.iloc[:, :] = np.nan
        for i in range(80, ret.shape[0]):
            # print(i)
            for j in range(ret.shape[1]):
                temp = [ret.iloc[i - 11, j], ret.iloc[i - 23, j], ret.iloc[i - 35, j], ret.iloc[i - 47, j],
                        ret.iloc[i - 59, j]]
                if np.sum(~np.isnan(temp)) >= min_year_num:
                    temp = np.nanmean(temp)
                else:
                    temp = np.nan
                result.iloc[i, j] = temp
        return result

    # B.2 Accounting-Related Anomalies
    # B.2.1 Profitability (20)

    # B.2.1.1 Return on Equity (roe)
    def calc_roe(self):
        q_net_income = self.get_data('FS_Comins', 'B002000000')#净利润
        oq_lag_bv = self.get_data('FS_Combas', 'A003000000').shift(3)#所有者权益合计
        prefer = self.get_data('FS_Combas', 'A003112101').shift(3)#优先股
        prefer.fillna(0, inplace=True)
        roe = q_net_income / (oq_lag_bv-prefer)#ROE = 净利润 / (所有者权益 - 优先股)
        return roe

    # B.2.1.2 4-quarter Changes in Return on Equity (droe)
    def calc_droe(self):
        q_net_income = self.get_data('FS_Comins', 'B002000000')
        oq_lag_bv = self.get_data('FS_Combas', 'A003000000').shift(3)
        roe = q_net_income / oq_lag_bv
        droe = roe - roe.shift(12)
        return droe

    # B.2.1.3 Return on Assets (roa)
    def calc_roa(self):
        q_net_income = self.get_data('FS_Comins', 'B002000000')
        oq_lag_ta = self.get_data('FS_Combas', 'A001000000').shift(3)#资产总计
        roa = q_net_income / oq_lag_ta
        return roa

    # B.2.1.4 4-quarter Changes in Return on Assets (droa)
    def calc_droa(self):
        q_net_income = self.get_data('FS_Comins', 'B002000000')
        oq_lag_ta = self.get_data('FS_Combas', 'A001000000').shift(3)
        roa = q_net_income / oq_lag_ta
        droa = roa - roa.shift(12)
        return droa

    # B.2.1.6 Quarterly Return on Net Operating Assets, Profit Margin, Assets Turnover (rnaq, pmq, atoq)
    def calc_rna(self):
        op_in = self.get_data('FS_Comins', 'B001300000')
        cash = self.get_data('FS_Combas', 'A001101000').shift(3)
        short_inv = self.get_data('FS_Combas', 'A001109000').shift(3)
        short_debt = self.get_data('FS_Combas', 'A002100000').shift(3)
        long_debt = self.get_data('FS_Combas', 'A002206000').shift(3)
        min_in = self.get_data('FS_Combas', 'A003200000').shift(3)
        prefer = self.get_data('FS_Combas', 'A003112101').shift(3)
        prefer.fillna(0, inplace=True)
        noa = short_debt + long_debt + min_in + prefer - cash - short_inv
        rna = op_in / noa
        return rna

    def calc_pm(self):
        op_in = self.get_data('FS_Comins', 'B001300000')
        op_rev = self.get_data('FS_Comins', 'B001101000')
        pm = op_in / op_rev
        return pm

    def calc_ato(self):
        op_rev = self.get_data('FS_Comins', 'B001101000')
        cash = self.get_data('FS_Combas', 'A001101000').shift(3)
        short_inv = self.get_data('FS_Combas', 'A001109000').shift(3)
        short_debt = self.get_data('FS_Combas', 'A002100000').shift(3)
        long_debt = self.get_data('FS_Combas', 'A002206000').shift(3)
        min_in = self.get_data('FS_Combas', 'A003200000').shift(3)
        prefer = self.get_data('FS_Combas', 'A003112101').shift(3)
        prefer.fillna(0, inplace=True)

        noa = short_debt + long_debt + min_in + prefer - cash - short_inv
        ato = op_rev / noa
        return ato

    # B.2.1.8 Quarterly Capital Turnover (ctq)
    def calc_ct(self):
        sales = self.get_data('FS_Comins', 'B001101000')#营业收入
        oy_lag_ta = self.get_data('FS_Combas', 'A001000000').shift(3)
        ct = sales / oy_lag_ta
        return ct

    # B.2.1.9 Gross Profits to Assets (gpa)
    def calc_gpa(self):
        tr_cgs = self.get_data('FS_Comins', 'B001000000',fs_freq='y')
        ta = self.get_data('FS_Combas', 'A001000000',fs_freq='y')
        gpa = tr_cgs / ta
        return gpa


    # B.2.1.11 Quarterly Gross Profits to Lagged Assets (gplaq)
    def calc_gpla(self):
        tr_cgs = self.get_data('FS_Comins', 'B001000000')
        ta = self.get_data('FS_Combas', 'A001000000')
        gpla = tr_cgs / ta.shift(3)
        return gpla

    # B.2.1.12 Operating Profits to Equity (ope)
    def calc_ope(self):
        op_pro = self.get_data('FS_Comins', 'B001300000',fs_freq='y')
        total_eq = self.get_data('FS_Combas', 'A003000000',fs_freq='y')
        prefer = self.get_data('FS_Combas', 'A003112101',fs_freq='y')
        prefer.fillna(0, inplace=True)
        book_eq = total_eq - prefer
        ope = op_pro / book_eq

        return ope


    # B.2.1.14 Quarterly Operating Profits to Lagged Equity (opleq)
    def calc_ople(self):
        op_pro = self.get_data('FS_Comins', 'B001300000')
        total_eq = self.get_data('FS_Combas', 'A003000000')
        prefer = self.get_data('FS_Combas', 'A003112101')
        prefer.fillna(0, inplace=True)
        book_eq = total_eq - prefer.shift(3)
        ople = op_pro / book_eq

        return ople

    # B.2.1.15 Operating Profits to Assets (opa)
    def calc_opa(self):
        op_pro = self.get_data('FS_Comins', 'B001300000')
        ta = self.get_data('FS_Combas', 'A001000000')
        opa = op_pro / ta
        return opa


    # B.2.1.17 Quarterly Operating Profits to Lagged Assets (oplaq)
    def calc_opla(self):
        op_pro = self.get_data('FS_Comins', 'B001300000')
        ta = self.get_data('FS_Combas', 'A001000000')
        opla = op_pro / ta.shift(3)
        return opla


    # B.2.1.19 Quarterly Taxable Income to book Income (tbiq)
    def calc_tbi(self):
        pretax_income = self.get_data('FS_Comins', 'B001000000')
        net_income = self.get_data('FS_Comins', 'B002000000')
        tbiq = pretax_income / net_income
        return tbiq


    # B.2.1.21 Quarterly Book Leverage (blq)
    def calc_bl(self):
        ta = self.get_data('FS_Combas', 'A001000000')
        be = self.get_data('FS_Combas', 'A003000000')
        prefer = self.get_data('FS_Combas', 'A003112101')
        prefer.fillna(0, inplace=True)
        blq = ta / (be-prefer)
        return blq

    # B.2.1.22 Annual Sales Growth (sg)
    def calc_sg(self):
        sales = self.get_data('FS_Comins', 'B001101000',fs_freq='y')
        sales_lag = sales.shift(12)
        result = (sales - sales_lag) / sales_lag
        return result

    # B.2.1.23 Quarterly Sales Growth (sgq)
    def calc_sgq(self):
        q_sales = self.get_data('FS_Comins', 'B001101000')
        sgq = q_sales / q_sales.shift(12)
        return sgq


    # B.2.1.25 Quarterly Fundamental score (fq)
    def calc_Fscore(self):
        # froa
        q_net_income = self.get_data('FS_Comins', 'B002000000')
        oq_lag_ta = self.get_data('FS_Combas', 'A001000000')
        froa = q_net_income / oq_lag_ta.shift(3)
        for i in range(froa.shape[0]):
            for j in range(froa.shape[1]):
                if froa.iloc[i, j] < 0:
                    froa.iloc[i, j] = 0
                elif froa.iloc[i, j] > 0:
                    froa.iloc[i, j] = 1

        # fcfa
        op_cash_flow = self.get_data('FS_Comscfd', 'C001000000')
        fcfa = op_cash_flow / oq_lag_ta.shift(3)
        for i in range(fcfa.shape[0]):
            for j in range(fcfa.shape[1]):
                if fcfa.iloc[i, j] <= 0:
                    fcfa.iloc[i, j] = 0
                elif fcfa.iloc[i, j] > 0:
                    fcfa.iloc[i, j] = 1

        # fdroa
        fdroa = froa - froa.shift(12)
        for i in range(fdroa.shape[0]):
            for j in range(fdroa.shape[1]):
                if fdroa.iloc[i, j] <= 0:
                    fdroa.iloc[i, j] = 0
                elif fdroa.iloc[i, j] > 0:
                    fdroa.iloc[i, j] = 1

        # fdlever
        long_debt = self.get_data('FS_Combas', 'A002206000')
        lever = long_debt / ((oq_lag_ta+oq_lag_ta.shift(3))/2)
        fdlever = lever - lever.shift(3)
        for i in range(fdlever.shape[0]):
            for j in range(fdlever.shape[1]):
                if fdlever.iloc[i, j] >= 0:
                    fdlever.iloc[i, j] = 0
                elif fdlever.iloc[i, j] < 0:
                    fdlever.iloc[i, j] = 1

        # fdliquid
        current_assets  = self.get_data('FS_Combas', 'A001100000')
        current_lia = self.get_data('FS_Combas', 'A002100000')
        current_ratio = current_assets / current_lia
        fdliquid = current_ratio - current_ratio.shift(3)
        for i in range(fdliquid.shape[0]):
            for j in range(fdliquid.shape[1]):
                if fdliquid.iloc[i, j] <= 0:
                    fdliquid.iloc[i, j] = 0
                elif fdliquid.iloc[i, j] > 0:
                    fdliquid.iloc[i, j] = 1

        # fdmargin
        operating_profit = self.get_data('FS_Comins', 'B001300000')
        sales = self.get_data('FS_Comins', 'B001101000')
        margin = operating_profit / sales
        fdmargin = margin - margin.shift(3)
        for i in range(fdmargin.shape[0]):
            for j in range(fdmargin.shape[1]):
                if fdmargin.iloc[i, j] <= 0:
                    fdmargin.iloc[i, j] = 0
                elif fdmargin.iloc[i, j] > 0:
                    fdmargin.iloc[i, j] = 1

        # fdturn
        op_rev = self.get_data('FS_Comins', 'B001101000')
        dturn = op_rev / oq_lag_ta
        fdturn = dturn - dturn.shift(3)
        for i in range(fdturn.shape[0]):
            for j in range(fdturn.shape[1]):
                if fdturn.iloc[i, j] <= 0:
                    fdturn.iloc[i, j] = 0
                elif fdturn.iloc[i, j] > 0:
                    fdturn.iloc[i, j] = 1

        Fscore = froa+fcfa+fdroa+fdlever+fdliquid+fdmargin+fdturn
        return Fscore


    # B.2.1.27 Quarterly O-score (oq)
    def calc_Oscore(self):
        total_assets = self.get_data('FS_Combas', 'A001000000')
        logta = total_assets.clip(lower=total_assets.quantile(0.01), upper=total_assets.quantile(0.99), axis=1)
        total_debt = self.get_data('FS_Combas', 'A002000000')
        tlta = total_assets/total_debt
        tlta = tlta.clip(lower=tlta.quantile(0.01), upper=tlta.quantile(0.99), axis=1)
        current_assets = self.get_data('FS_Combas', 'A001100000')
        current_lia = self.get_data('FS_Combas', 'A002100000')
        wcta = (current_assets-current_lia) / total_assets
        wcta = wcta.clip(lower=wcta.quantile(0.01), upper=wcta.quantile(0.99), axis=1)
        clca = current_lia / current_assets
        clca = clca.clip(lower=clca.quantile(0.01), upper=clca.quantile(0.99), axis=1)
        oeneg = total_debt - total_assets
        for i in range(oeneg.shape[0]):
            for j in range(oeneg.shape[1]):
                if oeneg.iloc[i, j] <= 0:
                    oeneg.iloc[i, j] = 0
                elif oeneg.iloc[i, j] > 0:
                    oeneg.iloc[i, j] = 1
        net_profit = self.get_data('FS_Comins', 'B002000000')
        nita = net_profit / total_assets
        nita = nita.clip(lower=nita.quantile(0.01), upper=nita.quantile(0.99), axis=1)
        earnings = self.get_data('FS_Comins', 'B002000000')
        tax = self.get_data('FS_Comins', 'B002100000')
        fund = earnings  + tax
        futl = fund / total_debt
        futl = futl.clip(lower=futl.quantile(0.01), upper=futl.quantile(0.99), axis=1)
        in2 = net_profit+net_profit.shift(3)
        for i in range(in2.shape[0]):
            for j in range(in2.shape[1]):
                if in2.iloc[i, j] >= 0:
                    in2.iloc[i, j] = 0
                elif in2.iloc[i, j] < 0:
                    in2.iloc[i, j] = 1
        chin = (net_profit-net_profit.shift(3))/(abs(net_profit)+abs(net_profit.shift(3)))

        Oscore = -1.32-0.407*logta + 6.03*tlta - 1.43*wcta + 0.076*clca - 1.72*oeneg - 2.37*nita - 1.83*futl + 0.285*in2- 0.521*chin
        return Oscore

    # B.2.2 Value (12)

    
         
    # B.2.2.3 Quarterly Book-to-Market Equity (bmq)
    def calc_bm(self):
        book_value = self.get_data('FS_Combas', 'A003000000')
        market_equity = self.get_data('TRD_Mnth', 'Msmvosd')
        market_equity = market_equity*1000
        prefer = self.get_data('FS_Combas', 'A003112101')
        prefer.fillna(0, inplace=True)
        bmq = (book_value-prefer) / market_equity
        return bmq


    # B.2.2.5 Quarterly Liabilities-to-Market Equity (dmq)
    def calc_dm(self):
        total_liabilities = self.get_data('FS_Combas', 'A002000000')
        market_equity = self.get_data('TRD_Mnth', 'Msmvosd')
        market_equity = market_equity*1000
        dmq = total_liabilities / market_equity
        return dmq

    # B.2.2.7 Quarterly Assets-to-Market Equity (amq)
    def calc_am(self):
        total_assets = self.get_data('FS_Combas', 'A001000000')
        market_equity = self.get_data('TRD_Mnth', 'Msmvosd')
        market_equity = market_equity*1000
        amq = total_assets / market_equity
        return amq

    # B.2.2.9 Quarterly Earnings-to-Price Ratio (epq)
    def calc_ep(self):
        net_profit = self.get_data('FS_Comins', 'B002000000')
        market_equity = self.get_data('TRD_Mnth', 'Msmvosd')
        market_equity = market_equity*1000
        epq = net_profit / market_equity
        return epq

    # B.2.2.11 Quarterly Cash Flow to Price (cfpq)
    def calc_cfp(self):
        cash_flows = self.get_data('FS_Comscfd', 'C005000000')
        market_equity = self.get_data('TRD_Mnth', 'Msmvosd')
        market_equity = market_equity*1000
        cfpq = cash_flows / market_equity
        return cfpq

    # B.2.2.12 5-year Sales Growth Rank (sr)
    def calc_sr(self):
        sales = self.get_data('FS_Comins', 'B001101000',fs_freq='y')
        sr = (sales.shift(48)-sales.shift(60)) / sales.shift(60)
        for i in range(4,0,-1):
            sr += (6-i)*(sales.shift(i*12-12)-sales.shift(i*12)) / sales.shift(i*12)
        return sr

    # B.2.2.14 Quarterly Enterprise Multiple (emq)
    def calc_em(self):
        operating_revenue = self.get_data('FS_Comins', 'B001101000')
        market_equity = self.get_data('TRD_Mnth', 'Msmvosd')
        market_equity = market_equity*1000
        tdebt = self.get_data('FS_Combas', 'A002000000')
        prefer = self.get_data('FS_Combas', 'A003112101')
        prefer.fillna(0, inplace=True)
        short_inv = self.get_data('FS_Combas', 'A001109000')
        cash = self.get_data('FS_Combas', 'A001101000')
        enterprise_value = market_equity + tdebt + prefer - cash - short_inv
        emq = enterprise_value / operating_revenue
        return emq

    # B.2.2.16 Quarterly Sales-to-Price Ratio (spq)
    def calc_sp(self):
        operating_revenue = self.get_data('FS_Comins', 'B001101000')
        market_equity = self.get_data('TRD_Mnth', 'Msmvosd')
        market_equity = market_equity*1000
        spq = operating_revenue / market_equity
        return spq

    # B.2.2.18 Quarterly Operating Cash Flow to Price Ratio (ocfpq)
    def calc_ocfp(self):
        op_cash_flow = self.get_data('FS_Comscfd', 'C001000000')
        market_equity = self.get_data('TRD_Mnth', 'Msmvosd')
        market_equity = market_equity*1000
        ocfpq = op_cash_flow / market_equity
        return ocfpq

    # B.2.2.19 Liabilities-to-Book Equity (de)
    def calc_de(self):
        total_liabilities = self.get_data('FS_Combas', 'A002000000',fs_freq='y')
        total_shareholder_equity = self.get_data('FS_Combas', 'A003000000',fs_freq='y')
        prefer = self.get_data('FS_Combas', 'A003112101')
        prefer.fillna(0, inplace=True)
        de = total_liabilities / (total_shareholder_equity-prefer)
        return de

    # B.2.2.22 Quarterly Enterprise Book-to-Price (ebpq) and Quarterly Net Debt-to-Price (ndpq)
    def calc_ebp(self):
        total_shareholder_equity = self.get_data('FS_Combas', 'A003000000')
        prefer = self.get_data('FS_Combas', 'A003112101')
        prefer.fillna(0, inplace=True)
        book_equity = total_shareholder_equity - prefer
        market_equity = self.get_data('TRD_Mnth', 'Msmvosd')
        market_equity = market_equity*1000
        long_debt = self.get_data('FS_Combas', 'A002206000')
        short_debt = self.get_data('FS_Combas', 'A002100000')
        short_inv = self.get_data('FS_Combas', 'A001109000')
        cash = self.get_data('FS_Combas', 'A001101000')
        net_debt = long_debt + short_debt + prefer - cash - short_inv

        ebp = (net_debt + book_equity) / (net_debt + market_equity)
        return ebp

    def calc_ndp(self):
        total_shareholder_equity = self.get_data('FS_Combas', 'A003000000')
        prefer = self.get_data('FS_Combas', 'A003112101')
        prefer.fillna(0, inplace=True)
        book_equity = total_shareholder_equity - prefer
        market_equity = self.get_data('TRD_Mnth', 'Msmvosd')
        market_equity = market_equity*1000
        long_debt = self.get_data('FS_Combas', 'A002206000')
        short_debt = self.get_data('FS_Combas', 'A002100000')
        short_inv = self.get_data('FS_Combas', 'A001109000')
        cash = self.get_data('FS_Combas', 'A001101000')
        net_debt = long_debt + short_debt + prefer - cash - short_inv

        ndp = net_debt / market_equity
        return ndp


    # B.2.3 Investment (20)

    # B.2.3.3 Quarterly Investment-to-Assets (agq)
    def calc_ag(self):
        total_assets = self.get_data('FS_Combas', 'A001000000')
        agq = (total_assets - total_assets.shift(12)) / total_assets.shift(12)
        return agq

    # B.2.3.4 Changes in PPE and inventory-to-assets (dpia)
    def calc_dpia(self):
        fixed_assets = self.get_data('FS_Combas', 'A001212000')
        inventory = self.get_data('FS_Combas', 'A001123000')
        total_assets = self.get_data('FS_Combas', 'A001000000').shift(12)
        dpia = (fixed_assets + inventory) / total_assets

        return dpia

    # B.2.3.5 Net Operating Assets, Changes in Net Operating Assets (noa, dnoa)
    def calc_noa(self):
        total_assets = self.get_data('FS_Combas', 'A001000000').shift(12)
        cash = self.get_data('FS_Combas', 'A001101000')
        short_inv = self.get_data('FS_Combas', 'A001109000')
        short_debt = self.get_data('FS_Combas', 'A002100000')
        long_debt = self.get_data('FS_Combas', 'A002206000')
        min_in = self.get_data('FS_Combas', 'A003200000')
        prefer = self.get_data('FS_Combas', 'A003112101')
        prefer.fillna(0, inplace=True)
        noa = short_debt + long_debt + min_in + prefer - cash - short_inv
        noa = noa / total_assets
        return noa

    def calc_dnoa(self):
        total_assets = self.get_data('FS_Combas', 'A001000000').shift(12)
        cash = self.get_data('FS_Combas', 'A001101000')
        short_inv = self.get_data('FS_Combas', 'A001109000')
        short_debt = self.get_data('FS_Combas', 'A002100000')
        long_debt = self.get_data('FS_Combas', 'A002206000')
        min_in = self.get_data('FS_Combas', 'A003200000')
        prefer = self.get_data('FS_Combas', 'A003112101')
        prefer.fillna(0, inplace=True)
        noa = short_debt + long_debt + min_in + prefer - cash - short_inv

        result = (noa.shift(12) - noa) / total_assets
        return result

    # B.2.3.6 x-year Investment Growth (ig, 2ig, 3ig)
    def calc_ig(self):
        cap_exp = self.get_data('FS_Comscfd','C002006000')
        ig = (cap_exp-cap_exp.shift(12)) / cap_exp.shift(12)
        return ig

    # B.2.3.8 Composite Equity Issuance (cei)
    def calc_cei(self):
        market_equity = self.get_data('TRD_Mnth', 'Msmvosd')
        market_equity = market_equity*1000
        me_growth = np.log(market_equity / market_equity.shift(12))
        ret = self.monthly_ret.copy()
        result = 1
        for i in range(1, 12 + 1):
            temp_lag = ret.shift(i)
            result = result * (1 + temp_lag)
        result = result - 1
        cei = me_growth - np.log(result)

        return cei


    # B.2.3.9 Composite Debt Issuance (cdi)
    def calc_cdi(self):
        total_liabilities = self.get_data('FS_Combas', 'A002000000')
        cdi = np.log((total_liabilities - total_liabilities.shift(12)) / total_liabilities.shift(12))
        return cdi

    # B.2.3.10 Inventory Growth (ivg)
    def calc_ivg(self):
        inventory = self.get_data('FS_Combas', 'A001123000')
        ivg = (inventory - inventory.shift(12)) / inventory.shift(12)
        return ivg

    # B.2.3.11 Inventory Change (ivchg)
    def calc_ivchg(self):
        total_assets = self.get_data('FS_Combas', 'A001000000')
        inventory = self.get_data('FS_Combas', 'A001123000')
        ivchg = (inventory - inventory.shift(12)) / total_assets.shift(12)
        return ivchg

    # B.2.3.12 Operating Accruals (oacc)
    def calc_oacc(self):
        total_assets = self.get_data('FS_Combas', 'A001000000')
        net_profit = self.get_data('FS_Comins', 'B002000000')
        op_cash_flows = self.get_data('FS_Comscfd', 'C001000000')
        oacc = (net_profit - op_cash_flows) / total_assets.shift(12)
        return oacc

    # B.2.3.13 Total Accruals (tacc)  ???????????????
    def calc_tacc(self):
        t_profit = self.get_data('FS_Comins', 'B001000000')
        cash_flows = self.get_data('FS_Comscfd', 'C005000000')
        total_assets = self.get_data('FS_Combas', 'A001000000')
        acc = (t_profit - cash_flows) / total_assets.shift(12)
        return acc

    # B.2.3.14 Changes in Net Noncash Working Capital, Current Operating Assets, and Current Operating Liabilities (dwc, dcoa, dcol)
    def calc_dwc(self):
        total_assets = self.get_data('FS_Combas', 'A001000000')
        current_assets  = self.get_data('FS_Combas', 'A001100000')
        cash = self.get_data('FS_Combas', 'A001101000')
        short_inv = self.get_data('FS_Combas', 'A001109000')
        coa = current_assets - cash - short_inv
        current_lia = self.get_data('FS_Combas', 'A002100000')
        short_debt = self.get_data('FS_Combas', 'A002101000')
        col = current_lia - short_debt
        wc = coa - col
        dwc = (wc - wc.shift(12)) / total_assets.shift(12)

        return dwc

    def calc_dcoa(self):
        total_assets = self.get_data('FS_Combas', 'A001000000')
        current_assets = self.get_data('FS_Combas', 'A001100000')
        cash = self.get_data('FS_Combas', 'A001101000')
        short_inv = self.get_data('FS_Combas', 'A001109000')
        coa = current_assets - cash - short_inv

        dcoa = (coa - coa.shift(12)) / total_assets.shift(12)
        return dcoa

    def calc_dcol(self):
        total_assets = self.get_data('FS_Combas', 'A001000000')
        current_lia = self.get_data('FS_Combas', 'A002100000')
        short_debt = self.get_data('FS_Combas', 'A002101000')
        col = current_lia - short_debt
        dcol = (col - col.shift(12)) / total_assets.shift(12)
        return dcol

    # B.2.3.15 Changes in Net Noncurrent Operating Assets, in Noncurrent Operating Assets, in Noncurrent Operating Liabilities (dnco, dnca, dncl)
    def calc_dnco(self):
        total_assets = self.get_data('FS_Combas', 'A001000000')
        current_assets = self.get_data('FS_Combas', 'A001100000')
        long_inv = self.get_data('FS_Combas', 'A001207000')
        nca = total_assets-current_assets - long_inv
        total_liabilities = self.get_data('FS_Combas', 'A002000000')
        current_lia = self.get_data('FS_Combas', 'A002100000')
        long_debt = self.get_data('FS_Combas', 'A002201000')
        ncl = total_liabilities - current_lia - long_debt
        nco = nca - ncl
        dnco = (nco - nco.shift(12)) / total_assets.shift(12)
        return dnco

    def calc_dnca(self):
        total_assets = self.get_data('FS_Combas', 'A001000000')
        current_assets = self.get_data('FS_Combas', 'A001100000')
        long_inv = self.get_data('FS_Combas', 'A001207000')
        nca = total_assets - current_assets - long_inv
        dnca = (nca - nca.shift(12)) / total_assets.shift(12)

        return dnca

    def calc_dncl(self):
        total_assets = self.get_data('FS_Combas', 'A001000000')
        total_liabilities = self.get_data('FS_Combas', 'A002000000')
        current_lia = self.get_data('FS_Combas', 'A002100000')
        long_debt = self.get_data('FS_Combas', 'A002201000')
        ncl = total_liabilities - current_lia - long_debt
        dncl = (ncl - ncl.shift(12)) / total_assets.shift(12)

        return dncl

    # B.2.3.16 Changes in Net Financial Assets, in Short-Term Investments, in Long-Term
    # Investments, in Financial Liabilities, and in Book Equity (dfin, dsti, dlti, dfnl, dbe)
    def calc_dfin(self):
        total_assets = self.get_data('FS_Combas', 'A001000000')
        short_inv = self.get_data('FS_Combas', 'A001109000')
        long_inv = self.get_data('FS_Combas', 'A001207000')
        financial_assets = short_inv + long_inv
        short_debt = self.get_data('FS_Combas', 'A002101000')
        long_debt = self.get_data('FS_Combas', 'A002201000')
        prefer = self.get_data('FS_Combas', 'A003112101')
        prefer.fillna(0, inplace=True)
        financial_lia = short_debt + long_debt + prefer
        fin = financial_assets - financial_lia
        dfin = (fin - fin.shift(12)) / total_assets.shift(12)

        return dfin

    def calc_dbe(self):
        total_assets = self.get_data('FS_Combas', 'A001000000')
        total_shareholder_equity = self.get_data('FS_Combas', 'A003000000')
        dbe = (total_shareholder_equity - total_shareholder_equity.shift(12)) / total_assets.shift(12)

        return dbe

    # B.2.4 Others (11)
    # B.2.4.1 Advertising Expense-to-Market (adm)
    def calc_adm(self):
        sell_exp = self.get_data('FS_Comins', 'B001209000')
        market_equity = self.get_data('TRD_Mnth', 'Msmvosd')
        market_equity = market_equity*1000
        adm = sell_exp / market_equity
        return adm

    # B.2.4.2 Growth in Advertising Expense (gad)
    def calc_gad(self):
        sell_exp = self.get_data('FS_Comins', 'B001209000')
        gad = (sell_exp-sell_exp.shift(12)) / sell_exp.shift(12)
        return gad

    # B.2.4.4 Quarterly R&D Expense to Market Equity (rdmq)
    def calc_rdm(self):
        management_fee = self.get_data('FS_Comins', 'B001210000')
        market_equity = self.get_data('TRD_Mnth', 'Msmvosd')
        market_equity = market_equity*1000
        rdm = management_fee / market_equity
        return rdm


    # B.2.4.6 Quarterly R&D Expense to Sales (rdsq)
    def calc_rds(self):
        management_fee = self.get_data('FS_Comins', 'B001210000')
        sales = self.get_data('FS_Comins', 'B001101000')
        rds = management_fee / sales
        return rds


    # B.2.4.8 Quarterly Operating Leverage (olq)
    def calc_ol(self):
        operating_cost = self.get_data('FS_Comins', 'B001201000')
        total_assets = self.get_data('FS_Combas', 'A001000000')
        ol = operating_cost / total_assets
        return ol

    # B.2.4.9 Hiring Rate (hn)
    def calc_hn(self):
        mon_ret = self.monthly_ret.copy()
        mon_ret.iloc[:,:]=1
        employee = self.get_data('CG_Ybasic','Y0601b')
        employee = mon_ret*employee
        employee.fillna(method='ffill', inplace=True)
        hn = (employee - employee.shift(12)) / (0.5*employee + 0.5*employee.shift(12))
        hn.index.names = ['Trdmnt']
        return hn

    # B.2.4.10 Firm Age (age)
    def calc_age(self):
        listdate = self.get_data('CG_Co','ListedDate')
        #listdate.set_index('Stkcd', inplace=True)
        result = self.monthly_ret.copy()
        result.iloc[:, :] = np.nan
        #print(result.shape[0],result.shape[1],listdate.shape[0],listdate.shape[1])
        for i in listdate.index:
            if i not in result.iloc[1, :].index:
                listdate.drop(index=i,inplace=True)
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                time = (int(result.index[i][0:4])-int(listdate.iloc[j,0][0:4]))*12 + int(result.index[i][4:6])-int(listdate.iloc[j,0][4:6])
                if time>=0:
                    result.iloc[i,j]=time
                else:
                    result.iloc[i,j]=np.nan
        return result

    # B.2.4.11 % Change in Sales minus % Change in Inventory (dsi)
    def calc_dsi(self):
        sales = self.get_data('FS_Comins', 'B001101000')
        apc_sales = (sales - sales.shift(12)) / sales.shift(12)
        net_inventory = self.get_data('FS_Combas', 'A001123000')
        apc_net_inventory = (net_inventory - net_inventory.shift(12)) / net_inventory.shift(12)
        dsi = apc_sales - apc_net_inventory
        return dsi

    # B.2.4.12 % Change in Sales minus % Change in Accounts Receivable (dsa)
    def calc_dsa(self):
        sales = self.get_data('FS_Comins', 'B001101000')
        apc_sales = (sales - sales.shift(12)) / sales.shift(12)
        acc_rec = self.get_data('FS_Combas', 'A001111000')
        apc_acc_rec = (acc_rec - acc_rec.shift(12)) / acc_rec.shift(12)
        dsa = apc_sales - apc_acc_rec
        return dsa

    # B.2.4.13 % Change in Gross Margin minus % Change in Sales (dgs)
    def calc_dgs(self):
        sales = self.get_data('FS_Comins', 'B001101000')
        apc_sales = (sales - sales.shift(12)) / sales.shift(12)
        op = self.get_data('FS_Comins', 'B001300000')
        apc_op = (op - op.shift(12)) / op.shift(12)
        dgs = apc_op - apc_sales
        return dgs

    # B.2.4.14 % Change in Sales minus % Change in SG&A (dss)
    def calc_dss(self):
        sales = self.get_data('FS_Comins', 'B001101000')
        apc_sales = (sales - sales.shift(12)) / sales.shift(12)
        sell_exp = self.get_data('FS_Comins', 'B001209000')
        admin_exp = self.get_data('FS_Comins', 'B001210000')
        SGA = sell_exp+admin_exp
        apc_SGA = (SGA - SGA.shift(12)) / SGA.shift(12)
        dss = apc_sales - apc_SGA
        return dss

    # B.2.4.15 Effective Tax Rate (etr)
    def calc_etr(self):
        tax_exp = self.get_data('FS_Comins', 'B002100000')
        earnings = self.get_data('FS_Comins', 'B002000000')
        tax = self.get_data('FS_Comins', 'B002100000')
        EBT = earnings + tax
        net_profit = self.get_data('FS_Comins', 'B002000000')
        market_equity = self.get_data('TRD_Mnth', 'Msmvosd')
        market_equity = market_equity*1000
        eps = net_profit / market_equity
        deps = eps - eps.shift(12)
        te = tax_exp/EBT
        etr = (te - (te.shift(36)+te.shift(24)+te.shift(12))/3)*deps
        return etr

    # B.2.4.16 Labor Force Efficiency (lfe)
    def calc_lfe(self):
        sales = self.get_data('FS_Comins', 'B001101000')
        mon_ret = self.monthly_ret.copy()
        mon_ret.iloc[:, :] = 1
        employee = self.get_data('CG_Ybasic', 'Y0601b')
        employee = mon_ret * employee
        employee.fillna(method='ffill', inplace=True)
        lfe = (sales/employee - sales.shift(12)/employee.shift(12)) / (sales.shift(12)/employee.shift(12))
        lfe.index.names = ['Trdmnt']
        return lfe


    # B.2.4.18 Quarterly Tangibility (tanq)
    def calc_tan(self):
        cash = self.get_data('FS_Combas', 'A001101000')
        short_inv = self.get_data('FS_Combas', 'A001109000')
        acc_rec = self.get_data('FS_Combas', 'A001111000')
        inventory = self.get_data('FS_Combas', 'A001123000')
        fixed_assets = self.get_data('FS_Combas', 'A001212000')
        ta = self.get_data('FS_Combas', 'A001000000')
        tan = cash+short_inv+0.715*acc_rec+0.547*inventory+0.535*fixed_assets
        tan = tan / ta

        return tan

    # B.2.4.19 Cash Flow Volatility (vcf)
    def calc_vcf(self, month_num=60, min_month_num=20):
        cash_flows = self.get_data('FS_Comscfd', 'C005000000')
        result = cash_flows.rolling(month_num, min_periods=min_month_num).apply(
            lambda x: np.std(x) / np.mean(x) if np.mean(x) != 0 else np.nan)
        return result

    # B.2.4.20 Cash to Assets (cta)
    def calc_cta(self):
        ccq = self.get_data('FS_Combas', 'A001101000')
        ta = self.get_data('FS_Combas', 'A001000000')
        cta = ccq / ta
        return cta

    # B.2.4.22 Earnings Smoothness (esm)
    def calc_esm(self,month_num = 36,min_month_num=12):
        earnings = self.get_data('FS_Comins', 'B002000000')
        total_asset = self.get_data('FS_Combas', 'A001000000')
        oper_cash_flow = self.get_data('FS_Comscfd','C001000000')
        esm = (earnings/total_asset.shift(12)).rolling(month_num, min_periods=min_month_num).std()/(oper_cash_flow/total_asset.shift(12)).rolling(month_num, min_periods=min_month_num).std()
        return esm


    # B.2.4.26 Quarterly Asset Liquidity (alaq, almq)
    def calc_ala(self):
        cash = self.get_data('FS_Combas', 'A001101000')
        current_assets = self.get_data('FS_Combas', 'A001100000')
        short_inv = self.get_data('FS_Combas', 'A001109000')
        total_asset = self.get_data('FS_Combas', 'A001000000')
        goodwill = self.get_data('FS_Combas', 'A001220000')
        intangibles = self.get_data('FS_Combas', 'A001218000')
        al = cash + 0.75*(current_assets - cash)+0.5*(total_asset-current_assets-goodwill-intangibles)
        ala = al / total_asset.shift(3)
        return ala

    def calc_alm(self):
        cash = self.get_data('FS_Combas', 'A001101000')
        current_assets = self.get_data('FS_Combas', 'A001100000')
        short_inv = self.get_data('FS_Combas', 'A001109000')
        total_asset = self.get_data('FS_Combas', 'A001000000')
        goodwill = self.get_data('FS_Combas', 'A001220000')
        intangibles = self.get_data('FS_Combas', 'A001218000')
        al = cash + 0.75 * (current_assets - cash) + 0.5 * (total_asset - current_assets - goodwill - intangibles)
        market_equity = self.get_data('TRD_Mnth', 'Msmvosd')
        market_equity = market_equity*1000
        total_eq = self.get_data('FS_Combas', 'A003000000')
        prefer = self.get_data('FS_Combas', 'A003112101')
        prefer.fillna(0, inplace=True)
        book_eq = total_eq - prefer
        alm = al / (total_asset + market_equity - book_eq).shift(3)
        return alm

    # B.2.4.27 Standard Unexpected Earnings (sue)
    def calc_sue(self, change_qnum=4, std_qnum=8, std_min_qnum=6):
        raw_data = pd.read_csv(self.local_data_path + 'FS_Comins.csv')
        raw_data.drop(index=raw_data[raw_data.Typrep == 'B'].index, inplace=True)
        raw_data['type'] = raw_data.Accper.apply(lambda x: x[5:7] + x[8:10])
        raw_data.drop(index=raw_data[raw_data.type == '0101'].index, inplace=True)
        raw_data.Accper = raw_data.Accper.apply(lambda x: x[0:4] + x[5:7])
        raw_data = raw_data.pivot(index='Accper', columns='Stkcd', values='B002000000')

        earnings = raw_data.copy()
        for i in range(1, earnings.shape[0]):
            if earnings.index[i - 1][-2:] != '12':
                earnings.iloc[i, :] = raw_data.iloc[i, :] - raw_data.iloc[i - 1, :]
        earnings_lag = earnings.shift(change_qnum)
        change = earnings - earnings_lag
        std = earnings.copy()
        std.iloc[:, :] = np.nan
        for i in range(earnings.shape[0]):
            for j in range(earnings.shape[1]):
                temp = np.array(earnings.iloc[i - (std_qnum - 1):i + 1, j])
                not_nan_num = np.sum(~np.isnan(temp))

                if not_nan_num >= std_min_qnum:
                    temp = temp[~np.isnan(temp)]
                    temp = np.diff(temp)
                    std.iloc[i, j] = np.std(temp, ddof=1)
        result = change / std

        new_ind = []
        for i in range(result.shape[0]):
            temp = result.index[i]
            if temp[-2:] == '12':
                temp = str(int(temp[0:4]) + 1) + '05'
            elif temp[-2:] == '03':
                temp = temp[0:4] + '07'
            elif temp[-2:] == '06':
                temp = temp[0:4] + '10'
            elif temp[-2:] == '09':
                temp = str(int(temp[0:4]) + 1) + '01'
            new_ind.append(temp)
        result['new_ind'] = new_ind
        result.set_index('new_ind', inplace=True)

        universe = self.monthly_ret.copy()
        data = universe.copy()
        data.iloc[:, :] = np.nan
        data.loc[:, :] = result.loc[:, :]
        data.fillna(method='ffill', inplace=True)
        universe = np.isnan(universe)
        data[universe] = np.nan

        return data

    # B.2.4.28 Revenue Surprises (rs)
    def calc_rs(self, change_qnum=4, std_qnum=8, std_min_qnum=6):
        raw_data = pd.read_csv(self.local_data_path + 'FS_Comins.csv')
        raw_data.drop(index=raw_data[raw_data.Typrep == 'B'].index, inplace=True)
        raw_data['type'] = raw_data.Accper.apply(lambda x: x[5:7] + x[8:10])
        raw_data.drop(index=raw_data[raw_data.type == '0101'].index, inplace=True)
        raw_data.Accper = raw_data.Accper.apply(lambda x: x[0:4] + x[5:7])
        raw_data = raw_data.pivot(index='Accper', columns='Stkcd', values='B001100000')

        earnings = raw_data.copy()
        for i in range(1, earnings.shape[0]):
            if earnings.index[i - 1][-2:] != '12':
                earnings.iloc[i, :] = raw_data.iloc[i, :] - raw_data.iloc[i - 1, :]
        earnings_lag = earnings.shift(change_qnum)
        change = earnings - earnings_lag
        std = earnings.copy()
        std.iloc[:, :] = np.nan
        for i in range(earnings.shape[0]):
            for j in range(earnings.shape[1]):
                temp = np.array(earnings.iloc[i - (std_qnum - 1):i + 1, j])
                not_nan_num = np.sum(~np.isnan(temp))

                if not_nan_num >= std_min_qnum:
                    temp = temp[~np.isnan(temp)]
                    temp = np.diff(temp)
                    std.iloc[i, j] = np.std(temp, ddof=1)
        result = change / std

        new_ind = []
        for i in range(result.shape[0]):
            temp = result.index[i]
            if temp[-2:] == '12':
                temp = str(int(temp[0:4]) + 1) + '05'
            elif temp[-2:] == '03':
                temp = temp[0:4] + '07'
            elif temp[-2:] == '06':
                temp = temp[0:4] + '10'
            elif temp[-2:] == '09':
                temp = str(int(temp[0:4]) + 1) + '01'
            new_ind.append(temp)
        result['new_ind'] = new_ind
        result.set_index('new_ind', inplace=True)

        universe = self.monthly_ret.copy()
        data = universe.copy()
        data.iloc[:, :] = np.nan
        data.loc[:, :] = result.loc[:, :]
        data.fillna(method='ffill', inplace=True)
        universe = np.isnan(universe)
        data[universe] = np.nan

        return data

    # B.2.4.29 Tax Expense Surprises (tes)
    def calc_tes(self, change_qnum=4, std_qnum=8, std_min_qnum=6):
        raw_data = pd.read_csv(self.local_data_path + 'FS_Comins.csv')
        raw_data.drop(index=raw_data[raw_data.Typrep == 'B'].index, inplace=True)
        raw_data['type'] = raw_data.Accper.apply(lambda x: x[5:7] + x[8:10])
        raw_data.drop(index=raw_data[raw_data.type == '0101'].index, inplace=True)
        raw_data.Accper = raw_data.Accper.apply(lambda x: x[0:4] + x[5:7])
        raw_data = raw_data.pivot(index='Accper', columns='Stkcd', values='B001100000')

        earnings = raw_data.copy()
        for i in range(1, earnings.shape[0]):
            if earnings.index[i - 1][-2:] != '12':
                earnings.iloc[i, :] = raw_data.iloc[i, :] - raw_data.iloc[i - 1, :]
        earnings_lag = earnings.shift(change_qnum)
        result = (earnings - earnings_lag) / earnings_lag

        new_ind = []
        for i in range(result.shape[0]):
            temp = result.index[i]
            if temp[-2:] == '12':
                temp = str(int(temp[0:4]) + 1) + '05'
            elif temp[-2:] == '03':
                temp = temp[0:4] + '07'
            elif temp[-2:] == '06':
                temp = temp[0:4] + '10'
            elif temp[-2:] == '09':
                temp = str(int(temp[0:4]) + 1) + '01'
            new_ind.append(temp)
        result['new_ind'] = new_ind
        result.set_index('new_ind', inplace=True)

        universe = self.monthly_ret.copy()
        data = universe.copy()
        data.iloc[:, :] = np.nan
        data.loc[:, :] = result.loc[:, :]
        data.fillna(method='ffill', inplace=True)
        universe = np.isnan(universe)
        data[universe] = np.nan

        return data

    #A2ME（Assets to marketcap）##这个因子好像在1555-1561行已经出现过了
    def calc_am(self):
        total_assets = self.get_data('FS_Combas', 'A001000000')
        market_equity = self.get_data('TRD_Mnth', 'Msmvosd')
        market_equity = market_equity*1000
        amq = total_assets / market_equity
        return amq

    #AT（Total Assets）
    def calc_at(self):
        total_assets = self.get_data('FS_Combas', 'A001000000')
        return total_assets

    #LME（Size）
    def calc_lme(self):
        price = self.get_data('TRD_Mnth', 'Mclsprc')    # 收盘价
        shares_outstanding = self.get_data('TRD_Mnth', 'Msmvosd')  # 千元单位的市值 / Price = 股数 × 价格，所以直接用千元市值再乘1000
        lme = price * shares_outstanding * 1000      # 乘1000是因为数据单位是千元
        return lme

    #Beta（CAPM Beta）
    def calc_capm(self, min_obs_corr=720, min_obs_vol=120):
        # 获取数据
        daily_ret = self.get_data('TRD_Dalyr', 'Dretwd')  # 个股日收益率
        market_ret = self.get_data('TRD_Cndalym', 'Cdretwdos')  # 市场日收益率
        rf = self.get_data('TRD_Nrrate', 'Nrrdaydt')  # 无风险利率
        # 统一日期
        df = daily_ret.copy()
        df['Trddt'] = df.index
        df = df.reset_index(drop=True)
        market_ret = market_ret.set_index('Trddt')
        rf = rf.set_index('Trddt')
        # 构建超额收益
        excess_stock = daily_ret.sub(rf['Nrrdaydt'], axis=0)
        excess_market = market_ret['Cdretwdos'] - rf['Nrrdaydt']
        # 计算 beta
        beta = excess_stock.copy()#构建一个空表 
        beta.iloc[:, :] = np.nan
        for col in excess_stock.columns: #遍历每一只股票
            stock_r = excess_stock[col].dropna() #去掉空值
            market_r = excess_market.loc[stock_r.index].dropna()  #取出某只股票的超额收益并且去掉空值
            combined_index = stock_r.index.intersection(market_r.index)   # 找出“股票收益”和“市场收益”这两个时间序列共有的日期
            if len(combined_index) < min_obs_corr: continue  #如果共有的日期小于750天就跳过
                continue
            #再次确认每只股票的收益和市场收益是不是在相同日期上有值
            stock_r = stock_r.loc[combined_index]
            market_r = market_r.loc[combined_index]

            corr = stock_r.corr(market_r) #计算当前这只股票的超额收益率（stock_r）和市场超额收益率（market_r）之间的相关性（皮尔逊相关系数）
            std_stock = np.log1p(stock_r).rolling(min_obs_vol).std().iloc[-1]   # 计算当前这只股票的超额收益率（stock_r）在过去120天内的标准差；np.log1p(stock_r)取对数
            std_market = np.log1p(market_r).rolling(min_obs_vol).std().iloc[-1] 

            if not np.isnan(corr) and std_market != 0:  #确保相关系数有效且市场标准差不是0才能计算
                beta.loc[combined_index[-1], col] = corr * (std_stock / std_market)  #写入计算出的 CAPM beta 值

        # 月末对齐
        beta['Trdmnt'] = beta.index
        beta.Trdmnt = beta.Trdmnt.apply(lambda x: x[0:6])
        beta.drop_duplicates(subset=['Trdmnt'], keep='last', inplace=True)
        beta.set_index('Trdmnt', inplace=True)
        return beta

    #LTurnover（Turnover）这个因子好像已经出现在第244行到251行的代码块中，并且多加了滚动 + 对数平滑处理
    def calc_turnover(self): 
        volume = self.get_data('TRD_Mnth', 'Mnshrtrd')  # 每月交易股数
        mv = self.get_data('TRD_Mnth', 'Msmvosd')       # 每月流通市值
        price = self.get_data('TRD_Mnth', 'Mclsprc')    # 每月收盘价

        shrout = mv / price # 估算流通股数
        volume_lag = volume.shift(1)  # 计算上月交易量
        turnover = volume.shift(1) / shrount  # Turnover = 上月成交量 / 当月流通股数
        return turnover 

    #MktBeta（Market Beta）
    def calc_mktbeta(self, trading_day_num=120, min_day_num=40):
        # 获取数据：股票月度收益、市值、市场收益、无风险收益
        daily_ret = self.get_data('TRD_Dalyr', 'Dretwd')
        market_ret = self.get_data('TRD_Cndalym', 'Cdretwdos')
        rf = self.get_data('TRD_Nrrate', 'Nrrdaydt')

        # 计算市场超额回报率（市场回报 - 无风险利率）
        market_and_rf = market_ret.merge(rf, on=['Trddt']) #根据日期（Trddt）对齐数据，这样每行就同时包含了某一天的市场回报和无风险利率
        rf_values = np.array(market_and_rf.iloc[:, -1]) # 无风险利率
        x = np.array(market_and_rf.iloc[:, -2]) - rf_values  # 市场超额回报率
        x = sm.add_constant(x)  # 给市场回报率加上常数项

        # 初始化结果数据框
        result = daily_ret.copy()
        result.iloc[:, :] = np.nan

        for j in range(result.shape[1]):
            # 计算每只股票的超额回报率
            y = np.array(daily_ret.iloc[:, j]) - rf_values
            # 使用滚动回归模型来计算每只股票的 Beta
            model = RollingOLS(y, x, window=trading_day_num, min_nobs=min_day_num).fit()
            result.iloc[:, j] = model.params[:, -1]  # Beta 是回归系数中的最后一列

        # 格式化结果为每月的最后一天
        result['Trdmnt'] = list(result.index)
        result['Trdmnt'] = result['Trdmnt'].apply(lambda x: x[0:6])
        result.drop_duplicates(subset=['Trdmnt'], keep='last', inplace=True)
        result.set_index('Trdmnt', inplace=True)

        return result
        





# end class

# ##############################################################################
# # All charateristics

# 118 all charateristics
chars_list_all = ['size',  'turnm', 'turnq', 'turna', 'vturn', 'cvturn','abturn', 'dtvm','dtvq','dtva','vdtv','cvd','Ami',
                  'idvc', 'idvff', 'idvq','tv', 'idsff','idsq', 'idsc', 'ts', 'cs', 'dbeta', 'betafp', 'betadm', 'beta',
                  'm1', 'm11', 'm60', 'm6', 'm3', 'indmom', 'm24', 'mchg', 'im12', 'im6', '52w', 'mdr', 'pr','abr','season',
                  'roe', 'droe', 'roa', 'droa', 'rna','pm','ato','ct', 'gpa', 'gpla', 'ope','ople','opa','opla','tbi','bl', 'sg','sgq', 'Fscore','Oscore',
                  'bm','dm', 'am', 'ep', 'cfp','sr','em', 'sp', 'ocfp', 'de', 'ebp','ndp',
                  'ag', 'dpia','noa','dnoa','ig','cei','cdi', 'ivg','ivchg','oacc','tacc', 'dwc','dcoa','dcol','dnco','dnca','dncl','dfin','dbe',
                  'adm', 'gad', 'rdm', 'rds','ol', 'hn','age','dsi','dsa', 'dgs','dss','etr','lfe','tan','vcf', 'cta', 'esm','ala','alm','sue', 'rs', 'tes','mkt_beta','vmg_beta','smb_beta','pmo_beta']

# 42 trading charateristics
chars_list_trading = ['size',  'turnm', 'turnq', 'turna', 'vturn', 'cvturn','abturn', 'dtvm','dtvq','dtva','vdtv','cvd','Ami',
                      'idvc', 'idvff', 'idvq','tv', 'idsff','idsq', 'idsc', 'ts', 'cs', 'dbeta', 'betafp', 'betadm', 'beta',
                      'm1', 'm11', 'm9', 'm6', 'm3', 'indmom', 'm24', 'mchg', 'im12', 'im6', '52w', 'mdr', 'pr','abr','season']

# 67 quarterly charateristics
chars_list_quarter = ['roe', 'droe', 'roa', 'droa', 'rna','pm','ato','ct','gpla','ople' ,'opla', 'tbi', 'bl', 'sg', 'Fscore','Oscore',
                      'bm', 'dm', 'am', 'ep', 'cfp','em', 'sp', 'ocfp', 'ebp','ndp',
                      'ag', 'dpia','noa','dnoa','ig','cei','cdi', 'ivg','ivchg','oacc','tacc', 'dwc','dcoa','dcol','dnco','dnca','dncl','dfin','dbe',
                      'adm', 'gad', 'rdm', 'rds','ol', 'hn','age','dsi','dsa', 'dgs','dss','etr','lfe','tan', 'vcf', 'cta', 'esm','ala','alm','sue', 'rs', 'tes']

# 6 yearly charateristics
chars_list_year = ['gpa','ope','opa','sr','de','sg']

# 4 Chinese beta charateristics
chars_list_4beta = ['mkt_beta','vmg_beta','smb_beta','pmo_beta']

'''
B.1.1 Liquidity : 13
B.1.2 Risk : 13
B.1.3 Past Returns : 15
B.2.1 Profitability : 20
B.2.2 Value : 12
B.2.3 Investment : 19
B.2.4 Other Anomalies : 22
Chinese 4 beta : 4
total : 118
'''

# # to calculate
# # you can choose any characteristics you want to calculate
# characters_copy = chars_list_all
# cut = chars_list_4beta + ['abr'] + ['age'] + ['hn'] + ['lfe']
# #python list 有加法没有减法

# for char in cut:
#     characters_copy.remove(char)
# #去除掉中国版四因子
# chars_list = characters_copy

chars_list = ['size',  'turnm', 'turnq', 'turna']


import time
print(len(chars_list))
mytest = AShareMarket('local')
counter = 0
start_time = time.time()  # 记录开始时间
for char in chars_list[:]:
    loop_start_time = time.time()  # 记录单个循环开始时间
    function = getattr(mytest, 'calc_' + char)
    char_result = function()
    char_result.to_csv('csmar_data/' + char + '.csv')
    counter = counter + 1
    loop_end_time = time.time()  # 记录单个循环结束时间
    loop_time = loop_end_time - loop_start_time  # 计算单个循环时间
    total_time = loop_end_time - start_time  # 计算累计时间
    print(counter, char + ' completed', '  single loop time: %.2f' % (loop_time/60), ' mins', '  total time: %.2f' % (total_time/60), 'mins')

print("All loops completed")
