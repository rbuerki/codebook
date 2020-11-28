"""NOTE: The functions in this notebook work for 
calculations on proportions only!

LIST OF FUNCTIONS
-----------------
- calc_confidence_bounds_binomial: Compute lower and upper bounds for a defined 
  confidence level based on a random variable with binomial / normal distribution.
- calc_experiment_size: Compute minimum number of samples for each group needed 
  to achieve a desired power level for a given effect size.
- calc_invariant_population: Compute statistics to check if your random 
  population sizing is within the expected standard error for a 50/50 split.
- calc_experiment_results: Compute observed difference with it's lower and upper 
  bounds based on a defined conficence level.


"""

import numpy as np
import scipy.stats as stats


def calc_confidence_bounds_binomial(p, n, alpha=0.05):
    """Compute lower and upper bounds for a defined confidence level based on a
    random variable with binomial / normal distribution. 

    ARGUMENTS:
        p_null = estimated proportion of successes (float)
        n = number of samples (int)
        alpha = Type-I error rate (float, default=0.05)
    
    RETURNS:
        Output of lower and upper bound values of confidence interval.
    """

    se = np.sqrt((p * (1 - p)) / n)
    z = stats.norm.ppf(1- alpha / 2) # scipy calculates lower-tail as default
    m = z * se
    lower = p - m
    upper = p + m
    
    print("lower bound: {:0.4f}, upper bound: {:0.4f}".format(lower, upper))


def calc_experiment_size(p_null, p_alt, alpha=.05, beta=.20, two_tails=True):
    """ Compute minimum number of samples for each group needed to achieve a 
    desired power level for a given effect size.
    
    ARGUMENTS:
        p_null = base success rate under null hypothesis (float)
        p_alt = desired success rate to be detected (float)
        alpha = Type-I error rate (float, default=0.05)
        beta = Type-II error rate (float, default=0.20)
        two_tails = indictation if one or two sided test (bool, default=True)
    
    RETURNS:
        n_min = minimum number of samples required for each group
    """
    
    # Get necessary z-scores and standard deviations (@ 1 obs per group)
    if two_tails:
        z_null = stats.norm.ppf(1 - (alpha / 2))
    else:
        z_null = stats.norm.ppf(1 - alpha)
    z_alt  = stats.norm.ppf(beta)
    se_null = np.sqrt(p_null * (1-p_null) + p_null * (1-p_null))
    se_alt  = np.sqrt(p_null * (1-p_null) + p_alt  * (1-p_alt))
    
    # Compute and return minimum sample size
    p_diff = p_alt - p_null
    n = ((z_null*se_null - z_alt*se_alt) / p_diff) ** 2
    n_min = np.ceil(n)
    
    print("min number of samples per group to achieve desired power:", n_min)
    return n_min


def calc_invariant_population(n_exp, n_cont, alpha=0.05):
    """Compute statistics to check if your random population sizing is within
    the expected standard error for a 50/50 split.

    ARGUMENTS:
        n_exp = number of samples in experiment group (int)
        n_cont = number of samples in control group (int)
        alpha = Type-I error rate (float, default=0.05)
    
    RETURNS:
        Output of test statistics. If the split is not within the expectations
        the function trows an assert error.
    """

    # calculate bounds of confidence intervall
    p = 0.5
    se = np.sqrt((p * (1 - p)) / (n_exp + n_cont))
    z = stats.norm.ppf(1- alpha / 2) # scipy calculates lower-tail by default
    m = z * se
    lower = p - m
    upper = p + m

    # calculate test statistic and p-value
    p_exp = n_exp / (n_exp + n_cont)
    z_score = abs(p_exp - p) / se # set to abs for consistency
    p_value = 2-(stats.norm.cdf(z_score) * 2) # doubled for two-tailed test

    assert p_exp >= lower and p_exp <= upper, \
        "WARNING: observed difference in sizes is outside of expectations.\n" \
        "proportion exp: {:0.3f} not within lower bound : {:0.3f} and " \
        "upper bound: {:0.3f}\np-value: {:0.3f} > alpha: {:0.3f}" \
        .format(p_exp, lower, upper, p_value, alpha)

    print("OK: Observed difference in sizes is within expectations.\n" \
        "proportion exp: {:0.3f} within lower bound : {:0.3f} and upper " \
        "bound: {:0.3f}\np-value: {:0.3f} > alpha: {:0.3f}" \
        .format(p_exp, lower, upper, p_value, alpha))

        

def calc_experiment_results(x_exp, x_cont, n_exp, n_cont, alpha=0.05, 
        two_tails=True):
    """Compute observed difference with it's lower and upper bounds based on
    a defined conficence level. 

    ARGUMENTS:
        x_exp = number of successful events in experiment group
        x_cont = number of successful events in control group
        n_exp = number of samples in experiment group
        n_cont = number of samples in control group
        alpha = Type-I error rate (float, default=0.05)
        two_tails = indictation if one or two sided test (bool, default=True)

    RETURNS:
        Output of estimated effect with it's lower and upper bound values and
        test statistic and p_value.
    """
    
    p_pool = (x_exp + x_cont) / (n_exp + n_cont)
    se_pool = np.sqrt(p_pool * (1-p_pool) * ((1 / n_exp) +(1 / n_cont)))
    p_diff = (x_exp / n_exp) - (x_cont / n_cont)
    
    # calculate bounds of confidence intervall
    if two_tails:
        z = stats.norm.ppf(1 - (alpha / 2))
    else:
        z = stats.norm.ppf(1 - alpha)
    m = z * se_pool
    lower = p_diff - m
    upper = p_diff + m
    
    # calculate test statistic and p-value
    z_score = (p_diff) / se_pool
    if two_tails:
        p_value = 2-(stats.norm.cdf(z_score) * 2)
    else:
        p_value = 1-stats.norm.cdf(z_score)

    if p_value < alpha:
        sign = "<"
        message = "STATISTICALLY HO CAN BE REJECTED."
    else: 
        sign = ">="
        message = "STATISTICALLY HO CAN NOT BE REJECTED."

    print("Observed difference: {:0.4f} with lower bound: {:0.4f} " \
        "and upper bound: {:0.4f}\np-value: {:0.4f} {} alpha: {:0.3f} " \
        "(z-score: {:0.4f})\n{}".format(p_diff, lower, upper, p_value, sign, \
        alpha, z_score, message))
