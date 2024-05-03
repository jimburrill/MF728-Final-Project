# MF728-Final-Project

The goal of our project was to develop a “quantamental” relative value corporate bond investment strategy. Where we attempt to create a long-only portfolio that would be benchmarked against an investment grade bond index. With the goal of either beating the index from a total return or risk-adjusted perspective by identifying relative value from mispricing between the fundamentals of the underlying companies and where their bonds trade on the curve. This involved determining the relative ranks of a company's financial health in the investment grade corporate bond universe. An optimization function was used to construct a portfolio of bonds on a quarterly basis with the goal of duration matching the Bloomberg US Corporate Total Return index. We chose to duration match the index to hopefully isolate areas we thought we could find alpha. The strategy was meant to be more like smart beta than L/S relative value. The following sections address the methodologies utilized to build our investment algorithm.

Requirements:
matplotlib==3.8.4
numpy==1.26.4
pandas==2.2.2
patsy==0.5.6
scipy==1.13.0
statsmodels==0.14.2
