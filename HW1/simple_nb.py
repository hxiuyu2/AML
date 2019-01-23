# -*- coding: utf-8 -*-
"""Simple Demo of a 1-feature Naive Bayes.

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

import math

# Female and male heights
f=[60,66,70,63,65]
m=[65,66,73,75,65,72]

# Calculate p(female) & p(male)
p_female = 1.0*len(f)/(len(f)+len(m))
p_male = 1 - p_female

# Calculate mean and variance for female heights
f_mean = np.mean(f)
f_var = np.var(f)

# Calculate mean and variance for male heights
m_mean = np.mean(m)
m_var = np.var(m)

print ("p(female) = %f, p(height|female) ~ N(mean = %f, var = %f)" % (p_female, f_mean, f_var))
print ("p(male) = %f, p(height|male) ~ N(mean = %f, var = %f)" % (p_male, m_mean, m_var))

x = np.linspace(50.0, 90.0, 100)
plt.plot(x,norm.pdf(x, f_mean, np.sqrt(f_var)))
plt.plot(x,norm.pdf(x, m_mean, np.sqrt(m_var)), c='r')

plt.show()

height = 67
print( "log p(f|height) \u221D %f" % (np.log(norm.pdf(height, f_mean, np.sqrt(f_var))) + np.log(p_female)))
print( "log p(m|height) \u221D %f" % (np.log(norm.pdf(height, m_mean, np.sqrt(m_var))) + np.log(p_male)))

