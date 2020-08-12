# -*- coding: utf-8 -*-
# Name: Automata.py
# Author: Shao Shidong
# Date: 2020/7/28
# Version: Python 3.6 64-bit

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

investors = ["Institutional Investor","Ordinary Investors","Noise Investors"]
operations = ["Buy","Hold","Sell"]
buy = []; sell = []
price = [15]
Rseries = [0]

def Message():
  m = np.random.normal(loc=0.5, scale=1)
  return m

def Plot(a,b):
  fig=plt.figure(dpi=80,figsize=(20,30))
  ax=fig.add_subplot(2,1,1)
  ax.plot(range(len(a)),a ,color='r',label='Price')
  plt.legend(loc='upper left')
  ax=fig.add_subplot(2,1,2)
  ax.plot(range(len(b)),b ,color='r',label='Return Series')
  plt.legend(loc='upper left')
  plt.savefig("E:/Figure/figure")
  plt.show()


class GameOfLife(object):
  def __init__(self, cells_shape):
    self.cells = np.ones(cells_shape)
    self.cells_state = np.zeros(cells_shape)
    self.cells_return = np.zeros(cells_shape)
    real_width = cells_shape[0] - 2
    real_height = cells_shape[1] - 2
    count1 = 0; count2 = 0
    """
    随机生成23个机构投资者、5986个普通投资者和3991个噪声投资者
    """
    while count1 < 23:
      x1 = np.random.randint(1,101); y1 = np.random.randint(1,101)
      if self.cells_state[x1, y1] == 0:
        self.cells_state[x1, y1] = 1
        count1 += 1
    while count2 < 5986:
      x2 = np.random.randint(1,101); y2 = np.random.randint(1,101)
      if self.cells_state[x2, y2] == 0:
        self.cells_state[x2, y2] = 2
        count2 += 1
    for i in range(1,101):
      for j in range(1,101):
        if self.cells_state[i, j] == 0:
          self.cells_state[i, j] = 3
    """
    随机生成投资者的行为，买入用+100表示、持仓用0表示、卖出用-100表示
    """
    for i in range(0,102):
      for j in range(0,102):
        if i==0 or i==101 or j==0 or j==101: self.cells[i,j] = -101
        else:
          flag = np.random.randint(1,4)
          if flag == 1: self.cells[i,j] = 100
          elif flag == 2: self.cells[i,j] = 0
          elif flag == 3: self.cells[i,j] = -100
      
    self.period = 0


  def Find_max(self, x, y, flag, target):
    """
    函数功能：
    -------------------------------------
    找出坐标为(x,y)的元胞邻居中的target投资者的多数行为，并返回

    参数:
    ----------------------------------
    flag, target：1->机构投资者；2->普通投资者；3->噪声投资者
    count_buy：邻居中选择“买入”的投资者数量
    count_hold：邻居中选择“持有”的投资者数量
    count_sell：邻居中选择“卖出”的投资者数量
    Max_operation: 邻居中的多数选择
    """
    cells = self.cells
    cells_state = self.cells_state
    count_buy = 0; count_hold = 0; count_sell = 0
    if flag == 1:
      if x <= 5:
        tool_x1 = 1
        tool_x2 = x+6
      elif x >= 96:
        tool_x1 = x-5
        tool_x2 = 101
      else:
        tool_x1 = x-5
        tool_x2 = x+6
      for i in range(tool_x1, tool_x2):
        if i <= x: tool_j = abs(x-5-i)
        else: tool_j = abs(x+5-i)
        if y-tool_j <= 0:
          y = tool_j+1
        if y+tool_j >= 101:
          y = 100-tool_j
        for j in range(y-tool_j, y+tool_j+1):
          if i!=x and j!=y:
            if cells_state[i, j] == target:
              if cells[i, j] == 100: count_buy += 1
              elif cells[i, j] == 0: count_hold += 1
              elif cells[i, j] == -100: count_sell += 1
    else:
      if x == 1:
        tool_x1 = 1
        tool_x2 = x+2
      elif x == 100:
        tool_x1 = x-1
        tool_x2 = 101
      else:
        tool_x1 = x-1
        tool_x2 = x+1
      for i in range(tool_x1,tool_x2):
        if y == 1:
          if cells_state[i, y+1] == target:
            if cells[i, y+1] == 100: count_buy += 1
            elif cells[i, y+1] == 0: count_hold += 1
            elif cells[i, y+1] == -100: count_sell += 1
        elif y == 100:
          if cells_state[i, y-1] == target:
            if cells[i, y-1] == 100: count_buy += 1
            elif cells[i, y-1] == 0: count_hold += 1
            elif cells[i, y-1] == -100: count_sell += 1
        else:
          if cells_state[i, y+1] == target:
            if cells[i, y+1] == 100: count_buy += 1
            elif cells[i, y+1] == 0: count_hold += 1
            elif cells[i, y+1] == -100: count_sell += 1
          if cells_state[i, y-1] == target:
            if cells[i, y-1] == 100: count_buy += 1
            elif cells[i, y-1] == 0: count_hold += 1
            elif cells[i, y-1] == -100: count_sell += 1
      if cells_state[x-1, y] == target:
        if cells[x-1, y] == 100: count_buy += 1
        elif cells[x-1, y] == 0: count_hold += 1
        elif cells[x-1, y] == -100: count_sell += 1
      if cells_state[x+1, y] ==target:
        if cells[x+1, y] == 100: count_buy += 1
        elif cells[x+1, y] == 0: count_hold += 1
        elif cells[x+1, y] == -100: count_sell += 1  
        
    Max = count_buy
    Max_operation = operations[0]
    if Max < count_hold:
      Max = count_hold
      Max_operation = operations[1]
    if Max < count_sell:
      Max = count_hold
      Max_operation = operations[2]
    return Max_operation 

    
  def Count(self, x, y, flag, target):
    """
    函数功能:
    ----------------------------------
    计算邻居中存在target数字代表的投资者的数量，并返回

    参数:
    ----------------------------------
    flag, target：1->机构投资者；2->普通投资者；3->噪声投资者
    count：target代表的投资者的数量
    """
    count = 0
    if flag == 1:
      """对机构投资者邻居"""
      if x <= 5:
        tool_x1 = 1
        tool_x2 = x+6
      elif x >= 96:
        tool_x1 = x-5
        tool_x2 = 101
      else:
        tool_x1 = x-5
        tool_x2 = x+6
      for i in range(tool_x1, tool_x2):
        if i <= x: tool_j = abs(x-5-i)
        else: tool_j = abs(x+5-i)
        if y-tool_j <= 0: y = tool_j+1
        if y+tool_j >= 101: y = 100-tool_j
        for j in range(y-tool_j, y+tool_j+1):
          if i!=x and j!=y:
            if self.cells_state[i, j] == target:
              count += 1
      return count
    else:
      """"对普通和噪声投资者"""
      if x == 1:
        tool_x1 = 1
        tool_x2 = x+2
      elif x == 100:
        tool_x1 = x-1
        tool_x2 = 101
      else:
        tool_x1 = x-1
        tool_x2 = x+1
      for i in range(tool_x1,tool_x2):
        if y == 1:
          if self.cells_state[i, y+1] == target:
            count += 1
        elif y == 100:
          if self.cells_state[i, y-1] == target:
            count += 1
        else:
          if self.cells_state[i, y+1] == target:
            count += 1
          if self.cells_state[i, y-1] == target:
            count += 1
      if self.cells_state[x-1, y] == target or self.cells_state[x+1, y] ==target:
        count += 1
      return count


  def Max_return(self, x, y, flag, target):
    """
    函数功能:
    ----------------------------------
    计算机构投资者邻居中target数字代表的投资者的累计最大收益率，并返回其当期操作

    参数:
    ----------------------------------
    target：2->普通投资者；3->噪声投资者
    """
    count = 0
    Oper = []
    Oper_x = []
    Oper_y = []
    if flag == 1:
      """
      对机构投资者邻居
      """
      if x <= 5:
        tool_x1 = 1
        tool_x2 = x+6
      elif x >= 96:
        tool_x1 = x-5
        tool_x2 = 101
      else:
        tool_x1 = x-5
        tool_x2 = x+6
      for i in range(tool_x1, tool_x2):
        if i <= x: tool_j = abs(x-5-i)
        else: tool_j = abs(x+5-i)
        if y-tool_j <= 0: y = tool_j+1
        if y+tool_j >= 101: y = 100-tool_j
        for j in range(y-tool_j, y+tool_j+1):
          if i!=x and j!=y:
            if self.cells_state[i, j] == target:
              if self.cells_return[i, j] >= 15 and target == 2:
                count += 1
                Oper.append(self.cells[i, j])
                Oper_x.append(i)
                Oper_y.append(j)
              elif self.cells_return[i, j] >= 20 and target == 3:
                count += 1
                Oper.append(self.cells[i, j])
                Oper_x.append(i)
                Oper_y.append(j)
      if count >= 1:
        Max_return = 0
        for i in range(len(Oper)):
          if self.cells_return[Oper_x[i],Oper_y[i]] >= Max_return:
            Max_return = self.cells_return[Oper_x[i],Oper_y[i]]
            Max_i = Oper_x[i]; Max_j = Oper_y[i];
        return self.cells[Max_i,Max_j]
      else:
        return -1


  def Rule(self, x, y, message, flag):
    """
     函数功能:
    ----------------------------------
    设定三类投资者的模仿规则，并在不同情况下更新坐标为(x,y)的元胞状态

    参数:
    ----------------------------------
    flag：1->机构投资者；2->普通投资者；3->噪声投资者
    Message~N(0.5,1)：市场消息，超过0.75为利好消息，低于0.25为利空消息，其他无消息
    """
    cell = self.cells
    cells_state = self.cells_state
    count1 = self.Count(x,y,flag,1)
    count2 = self.Count(x,y,flag,2)
    count3 = self.Count(x,y,flag,3)
    if flag == 1:
      if message >= 0.75:
        if count1!=0:
          Op = self.Find_max(x,y,flag,1)
          if Op == "Buy":
            prob = np.random.random()
            p1 = np.random.uniform(0.5,1)
            if p1*(2-message)>=1:
              cell[x, y] = 100
            else:
              if prob>=0 and prob<= p1*(2-message):
                cell[x, y] = 100
              else:
                cell[x, y] = 0
          elif Op == "Sell":
            prob = np.random.random()
            p1 = np.random.uniform(0.5,1)
            if prob>=0 and prob<=p1*(1-message):
              cell[x, y] = -100
            elif prob>p1*(1-message) and prob<=0.5+7*(1-message)/8:
              cell[x, y] = 0
            else:
              cell[x, y] = 100
          elif Op == "Hold":
            prob = np.random.random()
            p1 = np.random.uniform(0.5,1)
            if prob>=0 and prob<=p1:
              cell[x, y] = 0
            elif prob>p1 and prob<=0.5+p1*(2-message)/4:
              cell[x, y] = -100
            else:
              cell[x, y] = 100
        else:
          standard_operation2 = self.Max_return(x, y, 1, 2)
          standard_operation3 = self.Max_return(x, y, 1, 3)
          if standard_operation2==-1 and standard_operation3==-1:
            prob = np.random.random()
            Op = self.Find_max(x,y,flag,2)
            if Op == "Buy":
              if prob>=0 and prob<message:
                cell[x, y] = 100
              else:
                cell[x, y] = 0
            elif Op == "Hold":
              if prob>=0 and prob<message:
                cell[x, y] = 0
              else:
                cell[x, y] = -100
            else:
              if prob>=0 and prob<message:
                cell[x, y] = 0
              elif prob>=message and prob<(1+message)/2:
                cell[x, y] = -100
              else:
                cell[x, y] = 100
          else:
            if standard_operation2 != -1:
              cell[x, y] = standard_operation2
            elif standard_operation3 != -1 and standard_operation2 == -1:
              cell[x, y] = standard_operation3
      elif message <= 0.25:
        if count1!=0:
          Op = self.Find_max(x,y,flag,1)
          if Op == "Sell":
            prob = np.random.random()
            p1 = np.random.uniform(0.5,1)
            if p1*(2-message)>=1:
              cell[x, y] = -100
            else:
              if prob>=0 and prob<= p1*(2-message):
                cell[x, y] = -100
              else:
                cell[x, y] = 0
          elif Op == "Buy":
            prob = np.random.random()
            p1 = np.random.uniform(0.5,1)
            if prob>=0 and prob<=p1*(1-message):
              cell[x, y] = 100
            elif prob>p1*(1-message) and prob<=0.5+7*(1-message)/8:
              cell[x, y] = 0
            else:
              cell[x, y] = -100
          elif Op == "Hold":
            prob = np.random.random()
            p1 = np.random.uniform(0.5,1)
            if prob>=0 and prob<=p1:
              cell[x, y] = 0
            elif prob>p1 and prob<=0.5+p1*(2-message)/4:
              cell[x, y] = 100
            else:
              cell[x, y] = -100
        else:
          standard_operation2 = self.Max_return(x, y, 1, 2)
          standard_operation3 = self.Max_return(x, y, 1, 3)
          if standard_operation2==-1 and standard_operation3==-1:
            prob = np.random.random()
            Op = self.Find_max(x,y,flag,2)
            if Op == "Buy":
              if prob>=0 and prob<message:
                cell[x, y] = -100
              else:
                cell[x, y] = 0
            elif Op == "Hold":
              if prob>=0 and prob<message:
                cell[x, y] = 0
              else:
                cell[x, y] = -100
            else:
              if prob>=0 and prob<message:
                cell[x, y] = 100
              elif prob>=message and prob<(1+message)/2:
                cell[x, y] = -100
              else:
                cell[x, y] = 0
          else:
            if standard_operation2 != -1:
              cell[x, y] = standard_operation2
            elif standard_operation3 != -1 and standard_operation2 == -1:
              cell[x, y] = standard_operation3  
      else:
        if count1!=0:
          Op = self.Find_max(x,y,flag,1)
          if Op == "Buy":
            prob = np.random.random()
            p1 = np.random.uniform(0.5,1)
            if prob>=0 and prob<p1:
              cell[x, y] = 100
            else:
              cell[x, y] = 0
          elif Op == "Sell":
            prob = np.random.random()
            p1 = np.random.uniform(0.5,1)
            if prob>=0 and prob<=p1:
              cell[x, y] = -100
            else:
              cell[x, y] = 0
          elif Op == "Hold":
            prob = np.random.random()
            p1 = np.random.uniform(0.5,1)
            if prob>=0 and prob<=p1:
              cell[x, y] = 0
            elif prob>p1 and prob<=(1+p1)/2:
              cell[x, y] = -100
            else:
              cell[x, y] = 100
        else:
          standard_operation2 = self.Max_return(x, y, 1, 2)
          standard_operation3 = self.Max_return(x, y, 1, 3)
          if standard_operation2==-1 and standard_operation3==-1:
            cell[x, y] = 0
          else:
            if standard_operation2 != -1:
              cell[x, y] = standard_operation2
            elif standard_operation3 != -1 and standard_operation2 == -1:
              cell[x, y] = standard_operation3  
 
    elif flag == 2:
      if message >= 0.75:
        if count1!=0:
          Op = self.Find_max(x,y,flag,1)
          if Op == "Buy":
            prob = np.random.random()
            p2 = np.random.uniform(0.5,1)
            if p2*(1+message)>=1:
              cell[x, y] = 100
            else:
              if prob>=0 and prob<= p2*(1+message):
                cell[x, y] = 100
              else:
                cell[x, y] = 0
          elif Op == "Sell":
            prob = np.random.random()
            p2 = np.random.uniform(0.5,1)
            if prob>=0 and prob<=p2*(1-message):
              cell[x, y] = -100
            else:
              cell[x, y] = 0
          elif Op == "Hold":
            prob = np.random.random()
            p2 = np.random.uniform(0.5,1)
            if prob>=0 and prob<=p2:
              cell[x, y] = 0
            elif prob>p2 and prob<=0.5+p2*(2-message)/4:
              cell[x, y] = -100
            else:
              cell[x, y] = 100
        elif count2 == 8:
          Op = self.Find_max(x,y,flag,2)
          if Op == "Buy":
            prob = np.random.random()
            p3 = np.random.uniform(0.25,1)
            if p3*(1+message) >= 1:
              cell[x, y] = 100
            else:
              if prob>=0 and prob<p3*(1+message):
                cell[x, y] = 100
              elif prob>=p3*(1+message) and prob<(2*p3+2*message*p3+1)/3:
                cell[x, y] = -100
              else:
                cell[x, y] = 0
          elif Op == "Hold":
            prob = np.random.random()
            p3 = np.random.uniform(0.25,1)
            if prob>=0 and prob<p3:
              cell[x, y] = 0
            elif prob>=p3 and prob<=0.5+0.5*p3-0.25*message*p3:
              cell[x, y] = -100
            else:
              cell[x, y] = 100
          else:
            prob = np.random.random()
            p3 = np.random.uniform(0.25,1)
            if prob>=0 and prob<p3*(1-message):
              cell[x, y] = -100
            elif prob>=p3*(1-message) and prob<(2*p3-2*message*p3+1)/3:
              cell[x, y] = 100
            else:
              cell[x, y] = 0
        elif count3 == 8:
          Op = self.Find_max(x,y,flag,3)
          if Op == "Buy":
            prob = np.random.random()
            p4 = np.random.uniform(0,0.5)
            if prob>=0 and prob<p4*(1+message):
              cell[x, y] = 100
            elif prob>=p4*(1+message) and prob<(p4+message*p4+1)/2:
              cell[x, y] = -100
            else:
              cell[x, y] = 0
          elif Op == "Hold":
            prob = np.random.random()
            p4 = np.random.uniform(0,0.5)
            if prob>=0 and prob<p4:
              cell[x, y] = 0
            elif prob>=p3 and prob<=0.5+0.5*p4-0.25*message*p4:
              cell[x, y] = -100
            else:
              cell[x, y] = 100
          else:
            prob = np.random.random()
            p4 = np.random.uniform(0,0.5)
            if prob>=0 and prob<p4*(1-message):
              cell[x, y] = -100
            elif prob>=p4*(1-message) and prob<(p4-message*p4+1)/2:
              cell[x, y] = 100
            else:
              cell[x, y] = 0
        else:
          Op = self.Find_max(x,y,flag,2)
          if Op == "Buy":
            prob = np.random.random()
            if prob>=0 and prob<message:
              cell[x, y] = 100
            else:
              cell[x, y] = 0
          elif Op == "Hold":
            prob = np.random.random()
            if prob>=0 and prob<message:
              cell[x, y] = 0
            else:
              cell[x, y] = -100
          else:
            prob = np.random.random()
            if prob>=0 and prob<message:
              cell[x, y] = 0
            elif prob>=message and prob<(message+1)/2:
              cell[x, y] = 100
            else:
              cell[x, y] = 100
      elif message <= 0.25:
        if count1!=0:
          Op = self.Find_max(x,y,flag,1)
          if Op == "Buy":
            prob = np.random.random()
            p2 = np.random.uniform(0.5,1)
            if prob>=0 and prob<=p2*(1-message):
              cell[x, y] = 100
            else:
              cell[x, y] = 0
          elif Op == "Sell":
            prob = np.random.random()
            p2 = np.random.uniform(0.5,1)
            if p2*(1+message) >= 1:
              cell[x, y] = -100
            else:
              if prob>=0 and prob<=p2*(1+message):
                cell[x, y] = -100
              else:
                cell[x, y] = 0
          elif Op == "Hold":
            prob = np.random.random()
            p2 = np.random.uniform(0.5,1)
            if prob>=0 and prob<=p2:
              cell[x, y] = 0
            elif prob>p2 and prob<=0.5+p2*(2-message)/4:
              cell[x, y] = 100
            else:
              cell[x, y] = -100
        elif count2 == 8:
          Op = self.Find_max(x,y,flag,2)
          if Op == "Buy":
            prob = np.random.random()
            p3 = np.random.uniform(0.25,1)
            if prob>=0 and prob<p3*(1-message):
              cell[x, y] = 100
            elif prob>=p3*(1-message) and prob<(1+2*(1-message)*p3)/3:
              cell[x, y] = -100
            else:
              cell[x, y] = 0
          elif Op == "Hold":
            prob = np.random.random()
            p3 = np.random.uniform(0.25,1)
            if prob>=0 and prob<p3:
              cell[x, y] = 0
            elif prob>=p3 and prob<0.5+p3*(2-message)/4:
              cell[x, y] = 100
            else:
              cell[x, y] = -100
          else:
            prob = np.random.random()
            p3 = np.random.uniform(0.25,1)
            if (1+message)*p3>=1:
              cell[x, y] = -100
            else:
              if prob>=0 and prob<(1+message)*p3:
                cell[x, y] = -100
              elif prob>=message and prob<(1+2*(1+message)*p3)/3:
                cell[x, y] = 100
              else:
                cell[x, y] = 0
        elif count3 == 8:
          Op = self.Find_max(x,y,flag,2)
          if Op == "Buy":
            prob = np.random.random()
            p4 = np.random.uniform(0,0.5)
            if prob>=0 and prob<p4*(1-message):
              cell[x, y] = 100
            elif prob>=p4*(1-message) and prob<(1+(1-message)*p4)/2:
              cell[x, y] = -100
            else:
              cell[x, y] = 0
          elif Op == "Hold":
            prob = np.random.random()
            p4 = np.random.uniform(0,0.5)
            if prob>=0 and prob<p4:
              cell[x, y] = 0
            elif prob>=p4 and prob<0.5+p4*(2-message)/4:
              cell[x, y] = 100
            else:
              cell[x, y] = -100
          else:
            prob = np.random.random()
            p4 = np.random.uniform(0,0.5)
            if (1+message)*p4>=1:
              cell[x, y] = -100
            else:
              if prob>=0 and prob<(1+message)*p4:
                cell[x, y] = -100
              elif prob>=message and prob<(1+(1+message)*p4)/2:
                cell[x, y] = 100
              else:
                cell[x, y] = 0
        else:
          Op = self.Find_max(x,y,flag,2)
          if Op == "Buy":
            prob = np.random.random()
            if prob>=0 and prob<message:
              cell[x, y] = 100
            else:
              cell[x, y] = 0
          elif Op == "Sell":
            prob = np.random.random()
            if prob>=0 and prob<message:
              cell[x, y] = 0
            else:
              cell[x, y] = -100
          else:
            prob = np.random.random()
            if prob>=0 and prob<message/2:
              cell[x, y] = 100
            elif prob>=message/2 and prob<message:
              cell[x, y] = -100
            else:
              cell[x, y] = 0
      else:
        if count1!=0:
          Op = self.Find_max(x,y,flag,1)
          if Op == "Buy":
            prob = np.random.random()
            p2 = np.random.uniform(0.5,1)
            if prob>=0 and prob<p2:
              cell[x, y] = 100
            else:
              cell[x, y] = 0
          elif Op == "Sell":
            prob = np.random.random()
            p2 = np.random.uniform(0.5,1)
            if prob>=0 and prob<=p2:
              cell[x, y] = -100
            else:
              cell[x, y] = 0
          else:
            prob = np.random.random()
            p2 = np.random.uniform(0.5,1)
            if prob>=0 and prob<=(1-p2)/2:
              cell[x, y] = 100
            elif prob>(1-p2)/2 and prob<=1-p2:
              cell[x, y] = -100
            else:
              cell[x, y] = 0
        elif count2 == 8:
          Op = self.Find_max(x,y,flag,2)
          if Op == "Buy":
            prob = np.random.random()
            p3 = np.random.uniform(0.25,1)
            if prob>=0 and prob<p3:
              cell[x, y] = 100
            elif prob>=p3 and prob<(1+2*p3)/3:
              cell[x, y] = -100
            else:
              cell[x, y] = 0
          elif Op == "Sell":
            prob = np.random.random()
            p3 = np.random.uniform(0.25,1)
            if prob>=0 and prob<p3:
              cell[x, y] = -100
            elif prob>=p3 and prob<(1+2*p3)/3:
              cell[x, y] = 100
            else:
              cell[x, y] = 0
          else:
            prob = np.random.random()
            p3 = np.random.uniform(0.25,1)
            if prob>=0 and prob<p3:
              cell[x, y] = 0
            elif prob>=p3 and prob<(1+p3)/2:
              cell[x, y] = 100
            else:
              cell[x, y] = -100
        elif count3 == 8:
          Op = self.Find_max(x,y,flag,3)
          if Op == "Buy":
            prob = np.random.random()
            p4 = np.random.uniform(0,0.5)
            if prob>=0 and prob<p4:
              cell[x, y] = 100
            elif prob>=p4 and prob<(1+p4)/2:
              cell[x, y] = -100
            else:
              cell[x, y] = 0
          elif Op == "Sell":
            prob = np.random.random()
            p4 = np.random.uniform(0,0.5)
            if prob>=0 and prob<p4:
              cell[x, y] = -100
            elif prob>=p4 and prob<(1+p4)/2:
              cell[x, y] = 100
            else:
              cell[x, y] = 0
          else:
            prob = np.random.random()
            p4 = np.random.uniform(0,0.5)
            if prob>=0 and prob<p4:
              cell[x, y] = 0
            elif prob>=p4 and prob<(1+p4)/2:
              cell[x, y] = 100
            else:
              cell[x, y] = -100
        else:
          Op = self.Find_max(x,y,flag,2)
          prob = np.random.random()
          if Op == "Buy":
            if prob>=0 and prob<2/3:
              cell[x, y] = 100
            else:
              cell[x, y] = 0
          elif Op == "Sell":
            if prob>=0 and prob<2/3:
              cell[x, y] = -100
            else:
              cell[x, y] = 0
          else:
            if prob>=0 and prob<0.3:
              cell[x, y] = 100
            elif prob>=0.3 and prob<0.6:
              cell[x, y] = -100
            else:
              cell[x, y] = 0

    elif flag == 3:
      if message >= 0.75:
        if count1!=0:
          Op = self.Find_max(x,y,flag,1)
          if Op == "Buy":
            prob = np.random.random()
            p5 = np.random.uniform(0.75,1)
            if p5*(1+message)>=1:
              cell[x, y] = 100
            else:
              if prob>=0 and prob<= p5*(1+message):
                cell[x, y] = 100
              else:
                cell[x, y] = 0
          elif Op == "Sell":
            prob = np.random.random()
            p5 = np.random.uniform(0.75,1)
            if (2-message)*p5>=1:
              cell[x, y] = -100
            else:
              if prob>=0 and prob<=p5*(2-message):
                cell[x, y] = -100
              else:
                cell[x, y] = 0
          elif Op == "Hold":
            prob = np.random.random()
            p5 = np.random.uniform(0.75,1)
            if prob>=0 and prob<=p5:
              cell[x, y] = 0
            elif prob>p5 and prob<=0.5+p5*(2-message)/4:
              cell[x, y] = -100
            else:
              cell[x, y] = 100
        elif count2 == 8:
          Op = self.Find_max(x,y,flag,2)
          if Op == "Buy":
            prob = np.random.random()
            p6 = np.random.uniform(0.5,1)
            if (1+message)*p6>=1:
              cell[x, y] = 100
            else:
              if prob>=0 and prob<(1+message)*p6:
                cell[x, y] = 100
              elif prob>=(1+message)*p6 and prob<(1+4*(1+message)*p6)/5:
                cell[x, y] = -100
              else:
                cell[x, y] = 0
          elif Op == "Sell":
            prob = np.random.random()
            p6 = np.random.uniform(0.5,1)
            if prob>=0 and prob<(1-message)*p6:
              cell[x, y] = -100
            elif prob>=(1-message)*p6 and prob<(1+4*p6*(1-message))/5:
              cell[x, y] = 100
            else:
              cell[x, y] = 0
          else:
            prob = np.random.random()
            p6 = np.random.uniform(0.5,1)
            if prob>=0 and prob<p6:
              cell[x, y] = 0
            elif prob>=p6 and prob<0.5+p6*(2-message)/4:
              cell[x, y] = -100
            else:
              cell[x, y] = 100
        elif count3 == 8:
          Op = self.Find_max(x,y,flag,3)
          if Op == "Buy":
            prob = np.random.random()
            p7 = np.random.uniform(0,0.75)
            if (1+message)*p7>=1:
              cell[x, y] = 100
            else:
              if prob>=0 and prob<(1+message)*p7:
                cell[x, y] = 100
              elif prob>=(1+message)*p7 and prob<(1+4*p7*(1+message))/5:
                cell[x, y] = 0
              else:
                cell[x, y] = -100
          elif Op == "Sell":
            prob = np.random.random()
            p7 = np.random.uniform(0,0.75)
            if prob>=0 and prob<(1-message)*p7:
              cell[x, y] = -100
            elif prob>=(1-message)*p7 and prob<(1+4*p7*(1-message))/5:
              cell[x, y] = 100
            else:
              cell[x, y] = 0
          else:
            prob = np.random.random()
            p7 = np.random.uniform(0,0.75)
            if prob>=0 and prob<p7:
              cell[x, y] = 0
            elif prob>=p7 and prob<0.5+p7*(2-message)/4:
              cell[x, y] = -100
            else:
              cell[x, y] = 100
        else:
          Op = self.Find_max(x,y,flag,2)
          prob = np.random.random()
          if Op == "Buy":
            if prob>=0 and prob<message:
              cell[x, y] = 100
            else:
              cell[x, y] = 0
          elif Op == "Sell":
            if prob>=0 and prob<message:
              cell[x, y] = 0
            else:
              cell[x, y] = -100
          else:
            if prob>=0 and prob<(1-message)/2:
              cell[x, y] = 100
            elif prob>=(1-message)/2 and prob<1-message:
              cell[x, y] = -100
            else:
              cell[x, y] = 0
        
      elif message <= 0.25:
        if count1!=0:
          Op = self.Find_max(x,y,flag,1)
          if Op == "Sell":
            prob = np.random.random()
            p5 = np.random.uniform(0.75,1)
            if p5*(1+message)>=1:
              cell[x, y] = -100
            else:
              if prob>=0 and prob<= p5*(1+message):
                cell[x, y] = -100
              else:
                cell[x, y] = 0
          elif Op == "Buy":
            prob = np.random.random()
            p5 = np.random.uniform(0.75,1)
            if prob>=0 and prob<=p5*(1-message):
              cell[x, y] = 100
            else:
              cell[x, y] = 0
          elif Op == "Hold":
            prob = np.random.random()
            p5 = np.random.uniform(0.75,1)
            if prob>=0 and prob<=p5:
              cell[x, y] = 0
            elif prob>p5 and prob<=0.5+p5*(2-message)/4:
              cell[x, y] = 100
            else:
              cell[x, y] = -100
        elif count2 == 8:
          Op = self.Find_max(x,y,flag,2)
          if Op == "Sell":
            prob = np.random.random()
            p6 = np.random.uniform(0.5,1)
            if (1+message)*p6>=1:
              cell[x, y] = -100
            else:
              if prob>=0 and prob<(1+message)*p6:
                cell[x, y] = -100
              elif prob>=(1+message)*p6 and prob<(1+4*(1+message)*p6)/5:
                cell[x, y] = 100
              else:
                cell[x, y] = 0
          elif Op == "Buy":
            prob = np.random.random()
            p6 = np.random.uniform(0.5,1)
            if prob>=0 and prob<(1-message)*p6:
              cell[x, y] = 100
            elif prob>=(1-message)*p6 and prob<(1+4*p6*(1-message))/5:
              cell[x, y] = -100
            else:
              cell[x, y] = 0
          else:
            prob = np.random.random()
            p6 = np.random.uniform(0.5,1)
            if prob>=0 and prob<p6:
              cell[x, y] = 0
            elif prob>=p6 and prob<0.5+p6*(2-message)/4:
              cell[x, y] = 100
            else:
              cell[x, y] = -100
        elif count3 == 8:
          Op = self.Find_max(x,y,flag,3)
          if Op == "Sell":
            prob = np.random.random()
            p7 = np.random.uniform(0,0.75)
            if (1+message)*p7>=1:
              cell[x, y] = -100
            else:
              if prob>=0 and prob<(1+message)*p7:
                cell[x, y] = -100
              elif prob>=(1+message)*p7 and prob<(1+2*p7*(1+message))/3:
                cell[x, y] = 100
              else:
                cell[x, y] = 0
          elif Op == "Buy":
            prob = np.random.random()
            p7 = np.random.uniform(0,0.75)
            if prob>=0 and prob<(1-message)*p7:
              cell[x, y] = 100
            elif prob>=(1-message)*p7 and prob<(1+2*p7*(1-message))/3:
              cell[x, y] = -100
            else:
              cell[x, y] = 0
          else:
            prob = np.random.random()
            p7 = np.random.uniform(0,0.75)
            if prob>=0 and prob<p7:
              cell[x, y] = 0
            elif prob>=p7 and prob<0.5+p7*(2-message)/4:
              cell[x, y] = 100
            else:
              cell[x, y] = -100
        else:
          Op = self.Find_max(x,y,flag,2)
          prob = np.random.random()
          if Op == "Buy":
            if prob>=0 and prob<message:
              cell[x, y] = 100
            else:
              cell[x, y] = 0
          elif Op == "Sell":
            if prob>=0 and prob<message:
              cell[x, y] = 0
            else:
              cell[x, y] = -100
          else:
            if prob>=0 and prob<message/2:
              cell[x, y] = 100
            elif prob>=message/2 and prob<message:
              cell[x, y] = -100
            else:
              cell[x, y] = 0

      else:
        if count1!=0:
          Op = self.Find_max(x,y,flag,1)
          if Op == "Buy":
            prob = np.random.random()
            p5 = np.random.uniform(0.75,1)
            if prob>=0 and prob<= p5:
              cell[x, y] = 100
            else:
              cell[x, y] = 0
          elif Op == "Sell":
            prob = np.random.random()
            p5 = np.random.uniform(0.75,1)
            if prob>=0 and prob<=p5:
              cell[x, y] = -100
            else:
              cell[x, y] = 0
          elif Op == "Hold":
            prob = np.random.random()
            p5 = np.random.uniform(0.75,1)
            if prob>=0 and prob<=p5:
              cell[x, y] = 0
            elif prob>p5 and prob<=0.5+0.5*p5:
              cell[x, y] = -100
            else:
              cell[x, y] = 100
        elif count2 == 8:
          Op = self.Find_max(x,y,flag,2)
          if Op == "Buy":
            prob = np.random.random()
            p6 = np.random.uniform(0.5,1)
            if prob>=0 and prob<p6:
              cell[x, y] = 100
            elif prob>=p6 and prob<(1+4*p6)/5:
              cell[x, y] = -100
            else:
              cell[x, y] = 0
          elif Op == "Sell":
            prob = np.random.random()
            p6 = np.random.uniform(0.5,1)
            if prob>=0 and prob<p6:
              cell[x, y] = -100
            elif prob>=p6 and prob<(1+4*p6)/5:
              cell[x, y] = 100
            else:
              cell[x, y] = 0
          else:
            prob = np.random.random()
            p6 = np.random.uniform(0.5,1)
            if prob>=0 and prob<p6:
              cell[x, y] = 0
            elif prob>=p6 and prob<0.5+p6*0.5:
              cell[x, y] = -100
            else:
              cell[x, y] = 100
        elif count3 == 8:
          Op = self.Find_max(x,y,flag,3)
          if Op == "Buy":
            prob = np.random.random()
            p7 = np.random.uniform(0,0.75)
            if prob>=0 and prob<p7:
              cell[x, y] = 100
            elif prob>=p7 and prob<0.5+0.5*p7:
              cell[x, y] = 0
            else:
              cell[x, y] = -100
          elif Op == "Sell":
            prob = np.random.random()
            p7 = np.random.uniform(0,0.75)
            if prob>=0 and prob<p7:
              cell[x, y] = -100
            elif prob>=p7 and prob<0.5+0.5*p7:
              cell[x, y] = 100
            else:
              cell[x, y] = 0
          else:
            prob = np.random.random()
            p7 = np.random.uniform(0,0.75)
            if prob>=0 and prob<p7:
              cell[x, y] = 0
            elif prob>=p7 and prob<0.5+p7*0.5:
              cell[x, y] = -100
            else:
              cell[x, y] = 100
        else:
          Op = self.Find_max(x,y,flag,2)
          prob = np.random.random()
          if Op == "Buy":
            if prob>=0 and prob<0.8:
              cell[x, y] = 100
            else:
              cell[x, y] = 0
          elif Op == "Sell":
            if prob>=0 and prob<0.2:
              cell[x, y] = 0
            else:
              cell[x, y] = -100
          else:
            if prob>=0 and prob<0.2:
              cell[x, y] = 100
            elif prob>=0.2 and prob<0.4:
              cell[x, y] = -100
            else:
              cell[x, y] = 0
        
    return cell[x, y]

  
  def update_BHS(self,time):
    """
    函数功能：
    --------------------------------
    根据制定的模仿规则，更新每个元胞在time时刻的市场状态

    参数
    ----------
    time : 该时刻模拟市场轮次
    """
    cells = self.cells
    cells_state = self.cells_state
    cells_return = self.cells_return
    volume_buy = 0; volume_sell = 0
    currentR = 0
    BHS = np.zeros(self.cells.shape)
    for i in range(0, cells.shape[0]):
      for j in range(0, cells.shape[0]):
        BHS[i, j] = -101
    message = Message()
    k = np.random.normal(loc=0, scale=1)
    for i in range(1, cells.shape[0] - 1):
      for j in range(1, cells.shape[0] - 1):
        BHS[i, j] = self.Rule(i,j,message,cells_state[i, j])
        if BHS[i, j] == 100:
          volume_buy += 1
        elif BHS[i, j] == -100:
          volume_sell += 1
    if time != 0:
      p = (1+k*0.00001*(volume_buy-volume_sell))*price[time-1]
      price.append(p)
      currentR = 100*(math.log(price[time])-math.log(price[time-1]))
      Rseries.append(currentR)
    for i in range(1, cells.shape[0] - 1):
      for j in range(1, cells.shape[0] - 1):
        if BHS[i, j] == 100:
          cells_return[i ,j] -= currentR
        elif BHS[i, j] == -100:
          cells_return[i ,j] += currentR
    buy.append(volume_buy); sell.append(volume_sell)
    self.cells = BHS
    self.period += 1


  def update_Draw(self, N_round):
    """
    函数功能：
    --------------------------------
    1.在做图中画出每轮市场中投资者的行为，共N-round轮
    2.在右图中画出该次模拟市场中投资者的分布情况

    参数
    ----------
    N_round : 更新的轮数
    """
    plt.figure(figsize=(20,15))
    plt.ion()
    for i in range(N_round):
      plt.figure(figsize=(20,15))
      ax1 = plt.subplot(1,2,1)
      ax1.set_title('Round :{}'.format(self.period))
      """画出左图"""
      data = self.cells
      values = np.unique(data.ravel())
      values = values[1:]
      im = plt.imshow(self.cells)
      colors = [ im.cmap(im.norm(value)) for value in values]
      patches = [ mpatches.Patch(color=colors[i], label="{}".format(operations[i]) ) for i in range(len(values))]
      plt.legend(handles=patches, bbox_to_anchor=(1, 1), loc=2, borderaxespad=0. )
      """画出右图"""
      ax2 = plt.subplot(1,2,2)
      ax2.set_title('Investors')
      data2 = self.cells_state
      values2 = np.unique(data2.ravel())
      values2 = values2[1:]
      im2 = plt.imshow(self.cells_state)
      colors2 = [ im2.cmap(im2.norm(value)) for value in values2]
      patches2 = [ mpatches.Patch(color=colors2[i], label="{}".format(investors[i]) ) for i in range(len(values2))]
      plt.legend(handles=patches2, bbox_to_anchor=(1, 1), loc=2, borderaxespad=0. )
      plt.savefig("E:/Figure/Operations of {}".format(i))
      self.update_BHS(i)
      plt.pause(0.05)
    plt.ioff()


if __name__ == '__main__':
  game = GameOfLife(cells_shape=(102, 102))
  game.update_Draw(2000)
  Plot(price,Rseries)
  test=pd.DataFrame(data=Rseries)
  test.to_csv('E:/Figure/return.csv')
  print("提取完毕！")
