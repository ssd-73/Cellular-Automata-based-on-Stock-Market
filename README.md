I. Project Name: The Research of Herd Behavior in Stock Market Based on the Cellular Automata

II. Background: Herd Behavior shows that a large number of investors utilize the identical trading strategies or have the same preference for specific assets in a certain period of time, which can well explain the "bounded rationality" in the real world. Through the establishment of a cellular automata model with 90 imitative rules based on three categories of investors and three kinds of market messages, the cellular automata is applied for a simulation experiment on Herd Behavior in the real stock market, and proves that different types of investor behavior and different market messages will affect the behavior of investors. Based on the comparison of stock price, returns time series and investor behavior chart per round, we find that Herd Behavior will begin to highlight in the period of stock price acute fluctuations. Finally, a further statistical test is conducted on the variables such as the stock price and return rates generated by the simulation experiment.

III. Instructions:
1. The skeleton of this cellular automata contains several parts:
   (1) The map is 100×100. which means there are up to 10,000 investors.
   (2) Investors are divided into 3 categories, namely, institutional investors, ordinary investors, and noise investors.
   (3) There are 3 different categories of message in the market, obeying N(0, 1). If greater than 0.75, it is called "good news". If smaller          than 0.25, it is called "bad news". Otherwise, it is called "No news".
   (4) Based on three kinds of investors and three kinds of market message, there are up to 90 imitation rules.
2. How do u conduct this experiment by urself?
   (1) Modules requirement: math, numpy, pandas, and matplotlib
   (2) IDLE requirement: Python version >= 3
   (3) Use command line "Python Automata.py" in Linux, or open and run Automata.py directly in Windows
   (4) The folder called Figure contains the results of one experiment. If you wish, you can set any folder to save the results figures

IV. Suggestions:
1. I recommend u to test or run the codes in Jupyter notebook. If not, up to 2000 rounds can occupy about couples of hours to do so.
2. Because there are many random variables, the results of each simulatition will be different.
3. If u wanna use this code directly, please specify "@Author: Shao Shidong" in your Python files, and appendix of your paper, if possible.   
   Meanwhile, I still recommend u to code by urself.
