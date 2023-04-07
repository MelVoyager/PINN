class Lengendre:
    def P_0(self, x):
        return 1
    
    def P_0_prime(self, x):
        return 0
    
    def P_0_2prime(self, x):
        return 0
    
    def P_1(self, x):
        return x
    
    def P_1_prime(self, x):
        return 1
    
    def P_1_2prime(self, x):
        return 0
    
    def P_2(self, x):
        return 0.5 * (3 * x ** 2 - 1)
    
    def P_2_prime(self, x):
        return 3 * x
    
    def P_2_2prime(self, x):
        return 3
    
    def P_3(self, x):
        return 0.5 * (5 * x ** 3 - 3 * x)
    
    def P_3_prime(self, x):
        return 0.5 * (15 * x ** 2 - 3)
    
    def P_3_2prime(self, x):
         return 15 * x
    
    def P_4(self, x):
        return 0.125 * (35 * x ** 4 - 30 * x ** 2 + 3)
    
    def P_4_prime(self, x):
        return 0.5 * (35 * x ** 3 - 15 * x)
    
    def P_4_2prime(self, x):
        return 0.5 * (105 * x ** 2 - 15)
    
    def v(self, x, k=1):
        if k==1 :
            return self.P_2(x) - self.P_0(x)
        
        if k==2 :
            return self.P_3(x) - self.P_1(x)
        
        if k==3 :
            return self.P_4(x) - self.P_2(x)
        
    def v_prime(self, x, k=1):
         if k==1 : 
             return self.P_2_prime(x) - self.P_0_prime(x)
         
         if k==2 :
             return self.P_3_prime(x) - self.P_1_prime(x)
         
         if k==3 :
             return self.P_4_prime(x) - self.P_2_prime(x)
         
    def v_2prime(self, x, k=1):
         if k==1 : 
             return self.P_2_2prime(x) - self.P_0_2prime(x)
         
         if k==2 :
             return self.P_3_2prime(x) - self.P_1_2prime(x)
         
         if k==3 :
             return self.P_4_2prime(x) - self.P_2_2prime(x)