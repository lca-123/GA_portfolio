import numpy as np

# 使用滑动时间窗口进行投资组合选择
class Portfolio_Selection():
    """
    使用多期遗传算法+滚动时间窗口进行投资组合选择

    Paramters
    ----------
    data: numpy.ndarray
        股票对数收益率数据
    money: float, default 1000
        初始金额
    s: int, default 5
        遗传算法中的期数
    period: int, default 1
        持仓的持续时间，默认1个月
    t: int, default 60
        历史数据窗口期
    random_state: int or none
        随机数种子
    candidates: int or none
        候选解个数，若为空则默认30
    max_iter: int or none
        最大迭代次数，若为空则默认10 
    """

    def __init__(self,data,money=1000,s=5,t=60,period=1,
    random_state=None,candidates=30,max_iter=10):
        self.data=data
        self.money=money
        self.s=s
        self.t=t
        self.period=period
        self.random_state=random_state
        self.is_fit=False
        self.candidates=candidates
        self.max_iter=max_iter
    
    def fit(self):
        """"
        拟合模型
        """

        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        
        moneys = [self.money]
        exp_return=[]
        utility=[]
        weights=[]

        t1=self.t
        t2=len(self.data)-1

        for _t in range(t1,t2,self.period):
            # 划分数据集
            his_data=self.data[range(_t-t1,_t)]
            return_data=self.data[range(_t,_t+self.period)]

            # 生成模型
            p=Portfolio_GA(
                his_data=his_data,
                return_data=return_data,
                money=moneys[-1],
                s=self.s,
                random_state=int(np.random.rand()*10000),
                candidates=self.candidates,
                max_iter=self.max_iter)
            
            # 储存结果
            utility.append(p.ga())
            exp_return.append(p.cal_return())
            moneys.append(p.cal_money())
            weights.append(p.get_weights())
        
        self.is_fit=True

        self.moneys=moneys
        self.exp_return=exp_return
        self.utility=utility
        self.weights=weights
   
    def get_sharpe(self) -> float:
        """
        返回夏普比率
        """
        if self.is_fit==False:
            raise Exception('请先拟合模型')
        return np.mean(self.exp_return)/np.std(self.exp_return)
    
    def get_weights(self) -> list:
        """
        获得每一期的权重
        """
        return self.weights
    
    def cal_best_iter(self,max_max_iter=50) -> list:
        """
        根据论文方法，选取最佳的迭代次数
        还没实现

        随机选择一个时期的数据运行模型，画出期望效用和标准差图

        Paramters
        ----------
        max_max_iter: int, default 50
            迭代次数上限
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        t1=self.t
        t2=len(self.data)  

        _t=np.random.randint(t1,t2)

        his_data=self.data[range(_t-t1,_t)]
        return_data=self.data[range(_t,_t+self.period)]

        # 生成模型
        p=Portfolio_GA(
            his_data=his_data,
            return_data=return_data,
            money=self.money,
            s=self.s,
            random_state=int(np.random.rand()*10000),
            candidates=self.candidates,
            max_iter=max_max_iter)
        
        p.ga()
        return p.each_iter_exp_utility,p.each_iter_std
    
    def cal_best_candidates(self) -> list:
        """
        根据论文方法，选取最佳的候选解
        还没实现

        随机选择一个时期的数据运行模型，画出期望效用图
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        t1=self.t
        t2=len(self.data)  

        _t=np.random.randint(t1,t2)

    def update_parmaters(self,candidates=None,max_iter=None):
        """
        更新参数
        """
        if candidates is not None:
            self.candidates=candidates
        if max_iter is not None:
            self.max_iter=max_iter


    
# 使用遗传算法计算投资组合
class Portfolio_GA():
    """
    根据历史数据，使用多期遗传算法计算最佳投资组合

    Paramters
    ----------
    his_data: numpy.ndarray
        对数收益率
        股票历史数据，用于拟合
    return_data: numpy.ndarray
        对数收益率
        一个月后股票收益率，用于计算持仓收益
    money: int
        初始金额
    weights: list or none
        资产权重，若为空则默认等权
    random_state: int or none
        随机数种子，可为空
    candidates: int or none
        候选解个数，若为空则默认30
    max_iter: int or none
        最大迭代次数，若为空则默认10
    n_portfolio: int
        投资组合中资产个数，无需输入
    his_days: int
        历史数据期数,无需输入
    return_days: int
        预测数据期数，无需输入

    """

    def __init__(self,his_data,return_data,money,weights=None,s=5,
    random_state=None,candidates=30,max_iter=10):

        self.his_data=his_data
        self.his_days=len(his_data)
        self.n_portfolio=len(his_data[0])
        if weights is None:
            self.weights = np.array([1/self.n_portfolio]*self.n_portfolio)
        elif isinstance(weights,np.np.ndarray) is False:
            self.weights = np.array(weights)
        else:
            self.weights=weights
        # if len(return_data) != 1:
        #     raise Exception('return_data 输入错误，应输入未来一月的收益率情况')
        self.return_data=return_data
        self.return_days=len(return_data)
        self.money=money
        self.s=s
        self.random_state=random_state+1
        self.candidates=candidates
        self.max_iter=max_iter

    
    def cal_return(self) -> float:
        """
        计算投资组合持仓对应时期后的真实对数收益率(月化)

        Return
        ----------
        returns: float
            按照默认持仓一个月后的真实收益率
        """
        weight_money = self.weights*self.money
        each_assert_return = np.exp(np.sum(self.return_data,axis=0))
        total_assert = np.sum(weight_money*each_assert_return)
        total_assert_return = np.log(total_assert/self.money)
        return total_assert_return/len(self.return_data)  

    def cal_money(self,money=None) -> float:
        """
        计算投资组合持仓对应时期后的金额

        Paramters
        ----------
        money: float or none
            持仓金额，为空则取默认值
        
        Return
        --------
        money_return: float
            以默认持仓一个月后的金额
        """
        if money is None:
            money=self.money
        return money*np.exp(self.cal_return()*len(self.return_data))
    
    def get_weights(self):
        """
        返回现在的默认持仓
        """
        return list(self.weights)

    def _cal_ex_utility_(self,weights=None,s=None) -> float:
        """
        计算投资组合持仓的期望效用
        根据论文的方法，我们将历史数据随机分为s份，并计算其效用函数的均值

        Parmaters
        ----------
        weights: list, np.ndarray or none
            输入投资组合权重，若为空则使用类内权重
        s: int
            将历史数据分为s份

        Return
        ----------
        ex_utility: float
            期望效用
        """

        if weights is None:
            weights=self.weights

        if isinstance(weights,np.ndarray)==False:
            weights = np.array(weights)
        
        if self.random_state is not None:
            np.random.seed(self.random_state)

        if s is None:
             s=self.s
        
        # 暂时先平均分，后面可以尝试用dirichlet分布随机分
        # a = np.random.dirichlet([1]*s,1)*self.his_days

        # 各期的概率 暂定等权
        p = np.array([1/s]*s)

        # 计算各个时期的平均收益率
        mean_returns = []
        inc = self.his_days//s 
        for i in range(s):
            mean_return = list(np.mean(self.his_data[range(i*inc,(i+1)*inc)],axis=0))
            mean_returns.append(mean_return)
        
        # 计算各期投资组合收益率
        assert_each_return = np.sum(np.exp(np.array(mean_returns))*weights*self.money,axis=1)

        # 返回期望效用
        return np.sum(np.log(assert_each_return)*p)


    def ga(self):
        """
        使用遗传算法计算最佳投资组合

        Paramters
        ----------
        candidates: int or none
            候选解个数，默认30
        max_iter: int or none
            最大迭代次数，默认10
        
        Return
        ----------
        return: float
            遗传算法求解的投资组合持仓一个月后的真实对数收益率(月化)
        """

        if self.random_state is not None:
            np.random.seed(self.random_state)
        candidates=self.candidates
        max_iter=self.max_iter
        
        # 生成初始参数
        params=[np.random.dirichlet([1]*self.n_portfolio,1)[0] for i in range(candidates)]

        # 计算初始适应函数
        fitness_value=[]
        for i in range(candidates):
            param=params[i]
            return_p=self._cal_ex_utility_(param)
            fitness_value.append(return_p)
        
        # 保存每次迭代选取的最优解，以便选取最优参数
        each_iter_exp_utility=[]
        each_iter_std=[]

        # 迭代
        for iter in range(max_iter):
            # 适应度排序 从高到低两两交叉
            index=list(np.argsort(fitness_value))[::-1][:int(candidates/2)*2]

            # 选取表现最好的中间结果储存
            _index = np.argmax(fitness_value)
            each_iter_exp_utility.append(self._cal_ex_utility_(params[_index]))
            each_iter_std.append(np.std(np.array(fitness_value)[index]))
          

            for i in [i for i in range(int(candidates/2)*2)][::2]:
                new_param = (np.array(params[index[i]])+np.array(params[index[i+1]]))/2

                # 变异
                _i=np.random.uniform()
                if _i < 0.6667:
                    a,b = np.random.randint(0,self.n_portfolio-1,size=2)
                    while a == b:
                        b = np.random.randint(0,self.n_portfolio-1,size=1)

                    # 方法一 随机选取两个持仓交换
                    # new_param[a],new_param[b]=new_param[b],new_param[a]
                
                    # 方法二 随机选取两个持仓 将一个持仓的一部分加入另一个持仓
                    x=np.random.uniform(0,new_param[a]*0.9)
                    new_param[a]=new_param[a]-x
                    new_param[b]=new_param[b]+x


                new_param=list(new_param)
                params.append(new_param)
                return_p=self._cal_ex_utility_(new_param)
                fitness_value.append(return_p)
        
        self.each_iter_exp_utility=each_iter_exp_utility
        self.each_iter_std=each_iter_std
        
        # 选取表现最好的作为结果
        _index = np.argmax(fitness_value)
        self.weights=np.array(params[_index])
        return self.cal_return()
