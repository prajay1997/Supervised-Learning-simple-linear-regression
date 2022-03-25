########### Solution of linear Regression models ################
##################### Q1) weight vs Calaries Consumed ############

import pandas as pd      # for the dataframe and data manipulations 
import numpy as np      # for numerical calculations

# Load the data

data = pd.read_csv(r"C:\Users\praja\Desktop\Data Science\Supervised Learning Technique\Simple Linear Regression\datasets\calories_consumed.csv")

# to rename the column name 

data = data.rename(columns={'Weight gained (grams)':'weight_gain', 'Calories Consumed':"calaries_consumed"})


# EDA 
data.info()
data.describe()

#Graphical Representation

import matplotlib.pyplot as plt # for data visualizations 
plt.bar(height = data.calaries_consumed , x = np.arange(1,15,1)) # barplot
plt.hist(data.calaries_consumed)      # histogram
plt.boxplot(data.calaries_consumed)   # boxplot

plt.bar(height = data.weight_gain, x = np.arange(1,15,1))
plt.hist(data.weight_gain)
plt.boxplot(data.weight_gain)
 
# Scatter plot
plt.scatter(x = data["weight_gain"], y = data['calaries_consumed'])

# correlation
np.corrcoef(data.weight_gain, data.calaries_consumed) 

# covarience between two variable
cov_output = np.cov(data.weight_gain, data.calaries_consumed)[0,1]
cov_output
 
data.cov()

# import library
import statsmodels.formula.api as smf

# simple linear regression

model = smf.ols('calaries_consumed ~ weight_gain', data = data).fit()
model.summary()

pred1 = model.predict(pd.DataFrame(data['weight_gain']))

# Regression line 
plt.scatter(data.weight_gain, data.calaries_consumed)
plt.plot(data.weight_gain, pred1,"r")
plt.legend(["predicted line", "observed value"])
plt.show()

# Rmse value
res1 = data.calaries_consumed - pred1
res1_square = res1*res1
res1_mean = np.mean(res1_square)
rmse1 = np.sqrt(res1_mean)
rmse1

# Model building on Transformed Data

# Log transformation
 log = np.log(data['weight_gain'])

plt.scatter(x = log, y = data['calaries_consumed'], color = "green")
np.corrcoef(log, data['calaries_consumed'])

model1 = smf.ols('calaries_consumed ~ log', data = data).fit()
model1.summary()

pred2 = model1.predict(pd.DataFrame(data['weight_gain']))

# Regression line
plt.scatter(log, data.calaries_consumed)
plt.plot(log,pred2, "r")
plt.legend(["predicted line ","observed value"])
plt.show()

# RMSE calculation
res2 = data.calaries_consumed - pred2
res2_squ = res2*res2
res2_squ_mean = np.mean(res2_squ)
rmse2 = np.sqrt(res2_squ_mean)
rmse2

# exponential transformation
exp = np.log(data['calaries_consumed'])
plt.scatter(data.weight_gain, exp, color = "red")
np.corrcoef(data.weight_gain, exp)

model2 = smf.ols("exp ~ data.weight_gain", data = data).fit()
model2.summary()

pred3 = model2.predict(pd.DataFrame(data['weight_gain']))
pred3_cal = np.exp(pred3)
pred3_cal

# Regression line 
plt.scatter(data.weight_gain, exp)
plt.plot(data.weight_gain, pred3, "r")
plt.legend(["predicted line ","observed value"])
plt.show()

# RMSE calculation
res3 = data.calaries_consumed - pred3_cal
res3_squ = res3*res3 
res3_squ_mean = np.mean(res3_squ)
rmse3  = np.sqrt(res3_squ_mean)
rmse3

#### Polynomial transformation

model3 = smf.ols('np.log(calaries_consumed) ~ weight_gain + I(weight_gain*weight_gain)', data =data).fit()
model3.summary()

pred4 = model3.predict(pd.DataFrame(data))
pred4_cal = np.exp(pred4)
pred4_cal

# Regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X = data.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)
 
plt.scatter(data.weight_gain, np.log(data.calaries_consumed))
plt.plot(X , pred4_cal, "r")
plt.legend(["predicted line ","observed value"])
plt.show()

res4 = data.calaries_consumed - pred4_cal
res4_squ = res4*res4 
res4_squ_mean = np.mean(res4_squ)
rmse4  = np.sqrt(res4_squ_mean)
rmse4

# choose the best model using RMSE

data1 = {"Model":pd.Series(["SLR","log model","exp model","polynomial model"]), "RMSE":pd.Series([rmse1,rmse2, rmse3, rmse4])}
table_rmse = pd.DataFrame(data1)
table_rmse

###################
# The best model

from sklearn.model_selection import train_test_split

train, test = train_test_split(data,  test_size = 0.2)
plt.scatter(train.weight_gain, train.calaries_consumed, color ="black")

final_model = smf.ols('calaries_consumed ~ weight_gain', data =train).fit()
final_model.summary()

# predict on test data
test_pred = final_model.predict(pd.DataFrame(test))
test_pred


#  model evaluation on test data
res = test.calaries_consumed - test_pred
res_squ = res*res
res_squ_mean = np.mean(res_squ)
rmse_test = np.sqrt(res_squ_mean)
rmse_test 

# Prediction on train data
train_pred = final_model.predict(pd.DataFrame(train))
train_pred

# Regression line 
plt.scatter(train.weight_gain, train.calaries_consumed)
plt.plot(train.weight_gain, train_pred,"r")
plt.legend(["predicted line", "observed value"])
plt.show()


#  model evaluation on train data
res_train = train.calaries_consumed - train_pred
res_squ = res*res
res_squ_mean = np.mean(res_squ)
rmse_train = np.sqrt(res_squ_mean)
rmse_train

######################################################################

# solution on Simple Linear Regression Q2)###############

import pandas as pd
import numpy as np
# load the data 

dt = pd.read_csv(r"C:\Users\praja\Desktop\Data Science\Supervised Learning Technique\Simple Linear Regression\datasets\delivery_time.csv")
dt.info()
dt.describe()

dt = dt.rename(columns = {"Delivery Time":"delivery", "Sorting Time":"sort_time"})

# graphoical representation

import matplotlib.pyplot as plt
plt.bar(height = dt.sort_time, x= np.arange(1,22,1))
plt.hist(dt.sort_time)
plt.boxplot(dt.sort_time)

plt.bar(height = dt.delivery, x= np.arange(1,22,1))
plt.hist(dt.delivery)
plt.boxplot(dt.delivery)

# sctter plot
plt.scatter(x = dt.sort_time, y = dt.delivery, color = 'green')

# correlation coefficient r

np.corrcoef(dt.sort_time, dt.delivery)
# as r < 0.85 the data is moderately correlated

# covariance
cov_output = np.cov(dt.sort_time, dt.delivery)[0,1]
cov_output

dt.cov()

# import libraries
import statsmodels.formula.api as smf

# Simple Linear Expression
model = smf.ols('delivery~ sort_time', data =dt).fit()
model.summary()

pred1 = model.predict(pd.DataFrame(dt['sort_time']))
pred1
# regression Line 
plt.scatter(dt.sort_time, dt.delivery)
plt.plot(dt.sort_time, pred1, "r")
plt.legend(["predicted line","observed value"])
plt.show()

#RMSE value
res1 = dt.delivery- pred1
res1_square = res1*res1
res1_mean = np.mean(res1_square)
rmse1 = np.sqrt(res1_mean)
rmse1

# log Transformation

plt.scatter(x = np.log(dt.sort_time), y = dt.delivery, color = "green" )
np.corrcoef(np.log(dt.sort_time), dt.delivery)
# as r < 0.85 the data is moderately correlated

model1 = smf.ols('delivery~ np.log(sort_time)', data = dt).fit()
model1.summary()

pred2 = model1.predict(pd.DataFrame(dt['sort_time']))
pred2

# Regression Line
plt.scatter(np.log(dt.sort_time), dt.delivery)
plt.plot(np.log(dt.sort_time), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res2 = dt.delivery - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2

#### Exponential transformation ###

plt.scatter(x = dt.sort_time, y = np.log(dt.delivery), color = 'orange')
np.corrcoef(dt.sort_time, np.log(dt.delivery)) #correlation

model3 = smf.ols('np.log(delivery) ~ sort_time', data = dt).fit()
model3.summary()

pred3 = model3.predict(pd.DataFrame(dt['sort_time']))
pred3_dv = np.exp(pred3)
pred3_dv

# Regression Line
plt.scatter(dt.sort_time, np.log(dt.delivery))
plt.plot(dt.sort_time, pred3, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res3 = dt.delivery - pred3_dv
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3

#### Polynomial transformation ########

model4 = smf.ols('np.log(delivery) ~ sort_time + I(sort_time*sort_time)', data = dt).fit()
model4.summary()

pred4 = model4.predict(pd.DataFrame(dt))
pred4_dv = np.exp(pred4)
pred4_dv

# Regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X = dt.iloc[:, 1:].values
X_poly = poly_reg.fit_transform(X)
y = dt.iloc[:, 0].values


plt.scatter(dt.sort_time, np.log(dt.delivery))
plt.plot(X, pred4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()


# Error calculation
res4 = dt.delivery - pred4_dv
res_sqr4 = res4 * res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4


# Choose the best model using RMSE
data = {"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), "RMSE":pd.Series([rmse1, rmse2, rmse3, rmse4])}
table_rmse = pd.DataFrame(data)
table_rmse

###################
# The best model

from sklearn.model_selection import train_test_split

train, test = train_test_split(dt, test_size = 0.2)

finalmodel = smf.ols('np.log(delivery) ~ sort_time + I(sort_time*sort_time)', data = train).fit()
finalmodel.summary()

# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))
pred_test_dv = np.exp(test_pred)
pred_test_dv

# Model Evaluation on Test data
test_res = test.delivery - pred_test_dv
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse


# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))
pred_train_dv = np.exp(train_pred)
pred_train_dv

# Model Evaluation on train data
train_res = train.delivery - pred_train_dv
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse

##########################################################################

# solution on Simple Linear Regression Q3)###############

import pandas as pd
import numpy as np
# load the data 

hr= pd.read_csv(r"C:\Users\praja\Desktop\Data Science\Supervised Learning Technique\Simple Linear Regression\datasets\emp_data.csv")
hr.info()
hr.describe()

hr = hr.rename(columns = {"Salary_hike":"salary_hike", "Churn_out_rate":"churn"})

# graphoical representation

import matplotlib.pyplot as plt
plt.bar(height = hr.salary_hike, x= np.arange(1,11,1))
plt.hist(hr.salary_hike)
plt.boxplot(hr.salary_hike)

plt.bar(height = hr.churn, x= np.arange(1,11,1))
plt.hist(hr.churn)
plt.boxplot(hr.churn)

# sctter plot
plt.scatter(x = hr.salary_hike, y = hr.churn, color = 'green')

# correlation coefficient r

np.corrcoef(hr.salary_hike, hr.churn)
# as r > -0.85 the data is strongly correlated in the negative direction

# covariance
cov_output = np.cov(hr.salary_hike, hr.churn)[0,1]
cov_output

hr.cov()

# import libraries
import statsmodels.formula.api as smf

# Simple Linear Expression
model = smf.ols('churn~ salary_hike', data = hr).fit()
model.summary()

pred1 = model.predict(pd.DataFrame(hr['salary_hike']))
pred1
# regression Line 
plt.scatter(hr.salary_hike, hr.churn)
plt.plot(hr.salary_hike, pred1, "r")
plt.legend(["predicted line","observed value"])
plt.show()

#RMSE value
res1 = hr.churn - pred1
res1_square = res1*res1
res1_mean = np.mean(res1_square)
rmse1 = np.sqrt(res1_mean)
rmse1

# log Transformation

plt.scatter(x = np.log(hr.salary_hike), y = hr.churn, color = "green" )
np.corrcoef(np.log(hr.salary_hike), hr.churn)
# as r > -0.85 the data is strongly correlated in negative direction

model1 = smf.ols('churn ~ np.log(salary_hike)', data = hr).fit()
model1.summary()

pred2 = model1.predict(pd.DataFrame(hr['salary_hike']))
pred2

# Regression Line
plt.scatter(np.log(hr.salary_hike), hr.churn)
plt.plot(np.log(hr.salary_hike), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res2 = hr.churn - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2

#### Exponential transformation ###

plt.scatter(x = hr.salary_hike, y = np.log(hr.churn), color = 'orange')
np.corrcoef(hr.salary_hike, np.log(hr.churn)) #correlation

model3 = smf.ols('np.log(churn) ~ salary_hike', data = hr).fit()
model3.summary()

pred3 = model3.predict(pd.DataFrame(hr["salary_hike"]))
pred3_ch = np.exp(pred3)
pred3_ch

# Regression Line
plt.scatter(hr.salary_hike, np.log(hr.churn))
plt.plot(hr.salary_hike, pred3, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res3 = hr.churn - pred3_ch
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3

#### Polynomial transformation ########

model4 = smf.ols('np.log(churn) ~ salary_hike + I(salary_hike*salary_hike)', data = hr).fit()
model4.summary()

pred4 = model4.predict(pd.DataFrame(hr))
pred4_ch = np.exp(pred4)
pred4_ch

# Regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X = hr.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)
y = hr.iloc[:, 1:].values


plt.scatter(hr.salary_hike, np.log(hr.churn))
plt.plot(X, pred4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()


# Error calculation
res4 = hr.churn - pred4_ch
res_sqr4 = res4 * res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4


# Choose the best model using RMSE
data = {"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), "RMSE":pd.Series([rmse1, rmse2, rmse3, rmse4])}
table_rmse = pd.DataFrame(data)
table_rmse

###################
# The best model

from sklearn.model_selection import train_test_split

train, test = train_test_split(hr, test_size = 0.2)

finalmodel = smf.ols('np.log(churn) ~ salary_hike + I(salary_hike*salary_hike)', data = train).fit()
finalmodel.summary()

# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))
pred_test_ch = np.exp(test_pred)
pred_test_ch

# Model Evaluation on Test data
test_res = test.churn - pred_test_ch
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse


# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))
pred_train_ch = np.exp(train_pred)
pred_train_ch

# Model Evaluation on train data
train_res = train.churn - pred_train_ch
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse

############################################################################

# solution on Simple Linear Regression Q4)###############

import pandas as pd
import numpy as np
# load the data 

emp = pd.read_csv(r"C:\Users\praja\Desktop\Data Science\Supervised Learning Technique\Simple Linear Regression\datasets\Salary_Data.csv")
emp.info()
emp.describe()

emp = emp.rename(columns = {"YearsExperience":"exp", "Salary":"salary"})

# graphoical representation

import matplotlib.pyplot as plt
plt.bar(height = emp.exp, x= np.arange(1,31,1))
plt.hist(emp.exp)
plt.boxplot(emp.exp)

plt.bar(height = emp.salary, x= np.arange(1,31,1))
plt.hist(emp.salary)
plt.boxplot(emp.salary)

# sctter plot
plt.scatter(x = emp.exp, y = emp.salary, color = 'green')

# correlation coefficient r

np.corrcoef(emp.exp, emp.salary)
# as r > 0.85 the data is strongly correlated in the positive direction

# covariance
cov_output = np.cov(emp.exp, emp.salary)[0,1]
cov_output

emp.cov()

# import libraries
import statsmodels.formula.api as smf

# Simple Linear Expression
model = smf.ols('salary ~ exp', data = emp).fit()
model.summary()

pred1 = model.predict(pd.DataFrame(emp['exp']))
pred1
# regression Line 
plt.scatter(emp.exp, emp.salary)
plt.plot(emp.exp, pred1, "r")
plt.legend(["predicted line","observed value"])
plt.show()

#RMSE value
res1 = emp.salary - pred1
res1_square = res1*res1
res1_mean = np.mean(res1_square)
rmse1 = np.sqrt(res1_mean)
rmse1

# log Transformation

plt.scatter(x = np.log(emp.exp), y = emp.salary, color = "green" )
np.corrcoef(np.log(emp.exp), emp.salary)
# as r > 0.85 the data is strongly correlated in positive direction

model1 = smf.ols('salary ~ np.log(exp)', data = emp).fit()
model1.summary()

pred2 = model1.predict(pd.DataFrame(emp['exp']))
pred2

# Regression Line
plt.scatter(np.log(emp.exp), emp.salary)
plt.plot(np.log(emp.exp), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res2 = emp.salary - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2

#### Exponential transformation ###

plt.scatter(x = emp.exp, y = np.log(emp.salary), color = 'orange')
np.corrcoef(emp.exp, np.log(emp.salary)) #correlation

model3 = smf.ols('np.log(salary) ~ exp', data = emp).fit()
model3.summary()

pred3 = model3.predict(pd.DataFrame(emp["exp"]))
pred3_sal = np.exp(pred3)
pred3_sal

# Regression Line
plt.scatter(emp.exp, np.log(emp.salary))
plt.plot(emp.exp, pred3, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res3 = emp.salary - pred3_sal
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3

#### Polynomial transformation ########

model4 = smf.ols('np.log(salary) ~ exp + I(exp*exp) + I(exp*exp*exp)', data = emp).fit()
model4.summary()

pred4 = model4.predict(pd.DataFrame(emp))
pred4_sal = np.exp(pred4)
pred4_sal

# Regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3)
X = emp.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)
y = emp.iloc[:, 1:].values


plt.scatter(emp.exp, np.log(emp.salary))
plt.plot(X, pred4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()


# Error calculation
res4 = emp.salary - pred4_sal
res_sqr4 = res4 * res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4


# Choose the best model using RMSE
data = {"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), "RMSE":pd.Series([rmse1, rmse2, rmse3, rmse4])}
table_rmse = pd.DataFrame(data)
table_rmse

###################
# The best model

from sklearn.model_selection import train_test_split

train, test = train_test_split(emp, test_size = 0.2)

finalmodel = smf.ols('np.log(salary) ~ exp + I(exp*exp) + I(exp*exp*exp)', data = train).fit()
finalmodel.summary()

# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))
pred_test_sal = np.exp(test_pred)
pred_test_sal

# Model Evaluation on Test data
test_res = test.salary - pred_test_sal
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse


# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))
pred_train_sal = np.exp(train_pred)
pred_train_sal

# Model Evaluation on train data
train_res = train.salary - pred_train_sal
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse

#####################################################################
# solution on Simple Linear Regression Q5)###############

import pandas as pd
import numpy as np
# load the data 

gd = pd.read_csv(r"C:\Users\praja\Desktop\Data Science\Supervised Learning Technique\Simple Linear Regression\datasets\SAT_GPA.csv")
gd.info()
gd.describe()

gd = gd.rename(columns = {"SAT_Scores":"sat", "GPA":"gpa"})

# graphoical representation

import matplotlib.pyplot as plt
plt.bar(height = gd.sat, x= np.arange(1,201,1))
plt.hist(gd.sat)
plt.boxplot(gd.sat)

plt.bar(height = gd.gpa, x= np.arange(1,201,1))
plt.hist(gd.gpa)
plt.boxplot(gd.gpa)

# sctter plot
plt.scatter(x = gd.sat, y = gd.gpa, color = 'green')

# correlation coefficient r

np.corrcoef(gd.sat, gd.gpa)
# as r < 0.85 the data is not correlated 

# covariance
cov_output = np.cov(gd.sat, gd.gpa)[0,1]
cov_output

gd.cov()

# import libraries
import statsmodels.formula.api as smf

# Simple Linear Expression
model = smf.ols('gpa ~ sat', data = gd).fit()
model.summary()

pred1 = model.predict(pd.DataFrame(gd["sat"]))
pred1
# regression Line 
plt.scatter(gd.sat, gd.gpa)
plt.plot(gd.sat, pred1, "r")
plt.legend(["predicted line","observed value"])
plt.show()

#RMSE value
res1 = gd.gpa - pred1
res1_square = res1*res1
res1_mean = np.mean(res1_square)
rmse1 = np.sqrt(res1_mean)
rmse1

# log Transformation

plt.scatter(x = np.log(gd.sat), y = gd.gpa, color = "green" )
np.corrcoef(np.log(gd.sat), gd.gpa)
# as r < 0.85 the data is not correlated 

model1 = smf.ols('gpa ~ np.log(sat)', data = gd).fit()
model1.summary()

pred2 = model1.predict(pd.DataFrame(gd['sat']))
pred2

# Regression Line
plt.scatter(np.log(gd.sat), gd.gpa)
plt.plot(np.log(gd.sat), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res2 = gd.gpa - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2

#### Exponential transformation ###

plt.scatter(x = gd.sat, y = np.log(gd.gpa), color = 'orange')
np.corrcoef(gd.sat, np.log(gd.gpa)) #correlation

model3 = smf.ols('np.log(gpa) ~ sat', data = gd).fit()
model3.summary()

pred3 = model3.predict(pd.DataFrame(gd["sat"]))
pred3_gp = np.exp(pred3)
pred3_gp

# Regression Line
plt.scatter(gd.sat, np.log(gd.gpa))
plt.plot(gd.sat, pred3, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res3 = gd.gpa - pred3_gp
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3

#### Polynomial transformation ########

model4 = smf.ols('np.log(gpa) ~ sat + I(sat*sat) + I(sat*sat*sat) +I(sat*sat*sat*sat)+I(sat*sat*sat*sat*sat)+I(sat*sat*sat*sat*sat*sat)', data = gd).fit()
model4.summary()

pred4 = model4.predict(pd.DataFrame(gd))
pred4_gpa = np.exp(pred4)
pred4_gpa

# Regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 6)
X = gd.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)
y = gd.iloc[:, 1:].values


plt.scatter(gd.sat, np.log(gd.gpa))
plt.plot(X, pred4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()


# Error calculation
res4 = gd.gpa - pred4_gpa
res_sqr4 = res4 * res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4


# Choose the best model using RMSE
data = {"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), "RMSE":pd.Series([rmse1, rmse2, rmse3, rmse4])}
table_rmse = pd.DataFrame(data)
table_rmse

###################
# The best model

from sklearn.model_selection import train_test_split

train, test = train_test_split(gd, test_size = 0.2)

finalmodel = smf.ols('np.log(gpa) ~ sat + I(sat*sat) + I(sat*sat*sat) +I(sat*sat*sat*sat)+I(sat*sat*sat*sat*sat)+I(sat*sat*sat*sat*sat*sat)',  data = train).fit()
finalmodel.summary()

# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))
pred_test_gp = np.exp(test_pred)
pred_test_gp

# Model Evaluation on Test data
test_res = test.gpa - pred_test_gp
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse


# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))
pred_train_gp = np.exp(train_pred)
pred_train_gp

# Model Evaluation on train data
train_res = train.gpa - pred_train_gp
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse

##############################################################################

# solution on Simple Linear Regression Q5)###############

import pandas as pd
import numpy as np
# load the data 

gd = pd.read_csv(r"C:\Users\praja\Desktop\Data Science\Supervised Learning Technique\Simple Linear Regression\datasets\SAT_GPA.csv")
gd.info()
gd.describe()

gd = gd.rename(columns = {"SAT_Scores":"sat", "GPA":"gpa"})

# graphoical representation

import matplotlib.pyplot as plt
plt.bar(height = gd.sat, x= np.arange(1,201,1))
plt.hist(gd.sat)
plt.boxplot(gd.sat)

plt.bar(height = gd.gpa, x= np.arange(1,201,1))
plt.hist(gd.gpa)
plt.boxplot(gd.gpa)


# sctter plot
plt.scatter(x = gd.sat, y = gd.gpa, color = 'green')


# correlation coefficient r

np.corrcoef(gd.sat, gd.gpa)
# as r < 0.85 the data is not correlated 

# covariance
cov_output = np.cov(gd.sat, gd.gpa)[0,1]
cov_output

gd_std.cov()

# import libraries
import statsmodels.formula.api as smf

# Simple Linear Expression
model = smf.ols('gpa ~ sat', data = gd).fit()
model.summary()

pred1 = model.predict(pd.DataFrame(gd["sat"]))
pred1
# regression Line 
plt.scatter(gd.sat, gd.gpa)
plt.plot(gd.sat, pred1, "r")
plt.legend(["predicted line","observed value"])
plt.show()

#RMSE value
res1 = gd.gpa - pred1
res1_square = res1*res1
res1_mean = np.mean(res1_square)
rmse1 = np.sqrt(res1_mean)
rmse1

# log Transformation

plt.scatter(x = np.log(gd.sat), y = gd.gpa, color = "green" )
np.corrcoef(np.log(gd.sat), gd.gpa)
# as r < 0.85 the data is not correlated 

model1 = smf.ols('gpa ~ np.log(sat)', data = gd).fit()
model1.summary()

pred2 = model1.predict(pd.DataFrame(gd['sat']))
pred2

# Regression Line
plt.scatter(np.log(gd.sat), gd.gpa)
plt.plot(np.log(gd.sat), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res2 = gd.gpa - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2

#### Exponential transformation ###

plt.scatter(x = gd.sat, y = np.log(gd.gpa), color = 'orange')
np.corrcoef(gd.sat, np.log(gd.gpa)) #correlation

model3 = smf.ols('np.log(gpa) ~ sat', data = gd).fit()
model3.summary()

pred3 = model3.predict(pd.DataFrame(gd["sat"]))
pred3_gp = np.exp(pred3)
pred3_gp

# Regression Line
plt.scatter(gd.sat, np.log(gd.gpa))
plt.plot(gd.sat, pred3, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res3 = gd.gpa - pred3_gp
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3

#### Polynomial transformation ########

model4 = smf.ols('np.log(gpa) ~ sat + I(sat*sat) + I(sat*sat*sat) +I(sat*sat*sat*sat)+I(sat*sat*sat*sat*sat)+I(sat*sat*sat*sat*sat*sat)', data = gd).fit()
model4.summary()

pred4 = model4.predict(pd.DataFrame(gd))
pred4_gpa = np.exp(pred4)
pred4_gpa

# Regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 6)
X = gd.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)
y = gd.iloc[:, 1:].values


plt.scatter(gd.sat, np.log(gd.gpa))
plt.plot(X, pred4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()


# Error calculation
res4 = gd.gpa - pred4_gpa
res_sqr4 = res4 * res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4


# Choose the best model using RMSE
data = {"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), "RMSE":pd.Series([rmse1, rmse2, rmse3, rmse4])}
table_rmse = pd.DataFrame(data)
table_rmse

###################
# The best model

from sklearn.model_selection import train_test_split

train, test = train_test_split(gd, test_size = 0.2)

finalmodel = smf.ols('np.log(gpa) ~ sat + I(sat*sat) + I(sat*sat*sat) +I(sat*sat*sat*sat)+I(sat*sat*sat*sat*sat)+I(sat*sat*sat*sat*sat*sat)',  data = train).fit()
finalmodel.summary()

# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))
pred_test_gp = np.exp(test_pred)
pred_test_gp

# Model Evaluation on Test data
test_res = test.gpa - pred_test_gp
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse


# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))
pred_train_gp = np.exp(train_pred)
pred_train_gp

# Model Evaluation on train data
train_res = train.gpa - pred_train_gp
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse

##############################################################################


