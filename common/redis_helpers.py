class RedisKeys:
    def __init__(self, namespace):
        self.n = namespace
    
        self.trades = self._trades()
        self.wins = self._wins()
        self.buys = self._buys()
        self.sells = self._sells()
        self.holds = self._holds()
        self.actions = self._actions()
    
    def _trades(self):
        return self.n + '_trades'
        
    def _wins(self):
        return self.n + '_wins'
    
    def _buys(self):
        return self.n + '_buys'
        
    def _sells(self):
        return self.n + '_sells'
        
    def _holds(self):
        return self.n + '_holds'
        
    def _actions(self):
        return self.n + '_actions'
