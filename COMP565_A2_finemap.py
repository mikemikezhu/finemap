import pandas as pd
import numpy as np
from itertools import combinations
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy
import math

from abc import ABC, abstractmethod

# GitHub: https://github.com/mikemikezhu/finemap

"""
Constants
"""

DATA_PATH_LD = "data/LD.csv.gz"
DATA_PATH_Z_SCORE = "data/zscore.csv.gz"
DATA_PATH_SNP_PIP_REAL = "data/SNP_pip.csv.gz"

COMPRESSION_GZIP = "gzip"

CAUSAL_SNPS = ["rs10104559", "rs1365732", "rs12676370"]

CALCULATOR_TYPE_BAYES_FACTOR = "calculator_bayes_factor"
CALCULATOR_TYPE_PRIOR = "calculator_prior"
CALCULATOR_TYPE_POSTERIOR = "calculator_posterior"
CALCULATOR_TYPE_PIP = "calculator_pip"

"""
Calculator Factory
"""


class AbstractCalculator(ABC):

    @abstractmethod
    def calculate(self, **kwargs):
        raise NotImplementedError("Abstract method shall not be implemented")

    @abstractmethod
    def is_eligible(self, type: str) -> bool:
        raise NotImplementedError("Abstract method shall not be implemented")


class CalculatorFactory:

    def __init__(self) -> None:
        self._calculators = []
        children = AbstractCalculator.__subclasses__()
        if len(children) > 0:
            for child in children:
                self._calculators.append(child())

    def get_calculator(self, type: str) -> AbstractCalculator:

        if len(self._calculators) == 0:
            return None

        for calculator in self._calculators:
            if calculator.is_eligible(type):
                return calculator

        return None


"""
Q1: Bayes Factor
"""


class BayesFactorCalculator(AbstractCalculator):

    def calculate(self, **kwargs):

        result = None

        configs = kwargs.get("configs")
        ld = kwargs.get("ld")
        z_score = kwargs.get("z_score")
        silent = kwargs.get("silent") if kwargs.get(
            "silent") is not None else False

        assert configs is not None
        assert ld is not None
        assert z_score is not None

        # Z Score
        causal_z_scores = []
        for snp in configs:
            causal_z_score = (
                (z_score[z_score['Unnamed: 0'] == snp])["V1"]).to_numpy()
            causal_z_scores.append(causal_z_score)

        causal_z_scores = np.asarray(causal_z_scores)
        causal_z_scores = causal_z_scores[:, 0]
        if not silent:
            print("Causal Z Scores: \n", causal_z_scores)

        # RCC
        column_idx = [ld.columns.get_loc(c) for c in configs]
        row_idx = [id - 1 for id in column_idx]
        rcc = ld.iloc[row_idx, column_idx].to_numpy()
        np.fill_diagonal(rcc, 1.0)
        if not silent:
            print("RCC: \n", rcc)

        # RCC Star
        N = 498
        s_2 = 0.005
        sigma_cc = N * s_2 * np.identity(len(configs))  # Sigma_cc
        if not silent:
            print("Sigma CC: \n", sigma_cc)
        rcc_star = rcc + rcc @ sigma_cc @ rcc  # RCC Star = RCC + RCC @ Sigma_cc @ RCC
        if not silent:
            print("RCC Star: \n", rcc_star)

        # Bayes Factor
        mean = np.zeros(shape=(len(configs),))
        numerator = multivariate_normal.pdf(causal_z_scores,
                                            mean=mean,
                                            cov=rcc_star)
        denominator = multivariate_normal.pdf(causal_z_scores,
                                              mean=mean,
                                              cov=rcc)

        result = numerator / denominator
        return result

    def is_eligible(self, type: str) -> bool:
        return type == CALCULATOR_TYPE_BAYES_FACTOR


"""
Q2: Prior
"""


class PriorCalculator(AbstractCalculator):

    def calculate(self, **kwargs):

        m = kwargs.get("num_total_snps")
        k = kwargs.get("num_causal_snps")

        assert m is not None
        assert k is not None

        # Prior: (1 / m)^k * ((m - 1) / m)^(m - k)
        return (1 / m)**k * ((m - 1) / m)**(m - k)

    def is_eligible(self, type: str) -> bool:
        return type == CALCULATOR_TYPE_PRIOR


"""
Q3: Posterior
"""


class PosteriorCalculator(AbstractCalculator):

    def calculate(self, **kwargs):

        ld = kwargs.get("ld")
        z_score = kwargs.get("z_score")
        total_snps = kwargs.get("total_snps")
        max_causal_snps = kwargs.get("max_causal_snps")
        bayes_factor_calculator = kwargs.get("bayes_factor_calculator")
        prior_calculator = kwargs.get("prior_calculator")

        assert ld is not None
        assert z_score is not None
        assert total_snps is not None
        assert max_causal_snps is not None
        assert bayes_factor_calculator is not None
        assert prior_calculator is not None

        configs = []
        for num_causal in range(1, max_causal_snps + 1):
            configs += list(combinations(total_snps, num_causal))

        print("Total {} configs".format(len(configs)))

        columns = list(total_snps) + ['marginal', 'posterior', 'valid']
        configs_df = pd.DataFrame(np.zeros((len(configs), len(columns))),
                                  columns=columns)

        for i, c in enumerate(tqdm(configs)):

            c = list(c)

            # Bayes factor (Likelihood)
            pos_bf = bayes_factor_calculator.calculate(configs=c,
                                                       ld=ld,
                                                       z_score=z_score,
                                                       silent=True)
            # Prior
            pos_prior = prior_calculator.calculate(num_total_snps=len(total_snps),
                                                   num_causal_snps=len(c))

            # Mark the SNPs in the configuration
            configs_df.loc[i, c] = [1.0] * len(c)
            # Likelihood * prior
            configs_df.loc[i, 'marginal'] = pos_bf * pos_prior
            # Configuration is valid
            configs_df.loc[i, 'valid'] = 1.0

        # Remove invalid configurations
        configs_df = configs_df[configs_df["valid"] == 1.0]
        assert np.any(configs_df["valid"].to_numpy() == 0.0) == False

        print(configs_df.shape)

        # Calculate posteriors
        marginals = configs_df['marginal'].to_numpy()
        total_marginals = np.sum(marginals)
        posteriors = marginals / total_marginals
        configs_df.loc[:, 'posterior'] = posteriors

        print("Configs: {}\n", configs_df)
        print("Total marginals: ", total_marginals)
        print("Posteriors: ", posteriors)

        return posteriors, configs_df

    def is_eligible(self, type: str) -> bool:
        return type == CALCULATOR_TYPE_POSTERIOR


"""
Q4: PIP
"""


class PipCalculator(AbstractCalculator):

    def calculate(self, **kwargs):

        posteriors = kwargs.get("posteriors")
        configs_df = kwargs.get("configs_df")
        total_snps = kwargs.get("total_snps")

        assert posteriors is not None
        assert configs_df is not None
        assert total_snps is not None

        result = []

        for snp in total_snps:

            snp_df = configs_df[configs_df[snp] == 1.0]
            snp_posteriors = snp_df["posterior"].to_numpy()
            snp_pip = np.sum(snp_posteriors) / np.sum(posteriors)

            result.append(snp_pip)

        return result

    def is_eligible(self, type: str) -> bool:
        return type == CALCULATOR_TYPE_PIP


"""
Main
"""


def main():

    # Load data
    ld = pd.read_csv(DATA_PATH_LD, compression=COMPRESSION_GZIP)
    z_score = pd.read_csv(DATA_PATH_Z_SCORE, compression=COMPRESSION_GZIP)
    pip_real = pd.read_csv(DATA_PATH_SNP_PIP_REAL,
                           compression=COMPRESSION_GZIP)
    print("LD shape: {} \n{}".format(ld.shape, ld.head()))
    print("Z Score: {} \n{}".format(z_score.shape, z_score.head()))
    print("PIP real: {} \n{}".format(pip_real.shape, pip_real.head()))

    # Init calculator factory
    calculator_factory = CalculatorFactory()
    bayes_factor_calculator = calculator_factory.get_calculator(
        CALCULATOR_TYPE_BAYES_FACTOR)
    prior_calculator = calculator_factory.get_calculator(CALCULATOR_TYPE_PRIOR)

    # Calculate posterior
    posterior_calculator = calculator_factory.get_calculator(
        CALCULATOR_TYPE_POSTERIOR)
    total_snps = z_score.iloc[:, 0].to_numpy()
    num_total_snps = len(total_snps)
    posteriors, configs_df = posterior_calculator.calculate(ld=ld,
                                                            z_score=z_score,
                                                            total_snps=total_snps,
                                                            num_total_snps=num_total_snps,
                                                            max_causal_snps=3,  # Assume at maximum 3 causal SNPs
                                                            bayes_factor_calculator=bayes_factor_calculator,  # Dependency injection
                                                            prior_calculator=prior_calculator)  # Dependency injection
    # Plot
    sorted_posteriors = np.sort(posteriors)
    plt.clf()
    plt.scatter(np.arange(len(sorted_posteriors)), sorted_posteriors)
    plt.xlabel("Sorted configurations")
    plt.ylabel("Configuration posterior")
    plt.title("Posteriors of all of the valid configurations in increasing order")
    plt.grid(0.5)
    plt.savefig("posteriors.png")
    plt.close()

    # Caclulate PIP
    pip_calculator = calculator_factory.get_calculator(CALCULATOR_TYPE_PIP)
    pips = pip_calculator.calculate(posteriors=posteriors,
                                    configs_df=configs_df,
                                    total_snps=total_snps)
    pips = np.asarray(pips)
    print("PIP: ", pips)

    # Output inferred PIPs
    pips_df = pd.DataFrame(columns=["SNPs", "Inferred PIPs"])
    pips_df[pips_df.columns[0]] = total_snps
    pips_df[pips_df.columns[1]] = pips
    pips_df.to_csv("COMP565 A2 SNP pip.csv.gz", compression=COMPRESSION_GZIP)
    print(pips_df)

    # Plot
    causal_index = np.empty((len(CAUSAL_SNPS)), dtype=int)
    for i, snp in enumerate(CAUSAL_SNPS):
        index = np.argwhere((total_snps == snp))[0, 0]
        causal_index[i] = index

    total_snps_index = np.arange(len(pips))
    non_causal_index = np.delete(total_snps_index, obj=causal_index)

    causal_pips = pips[causal_index]
    non_causal_pips = pips[non_causal_index]

    # Two-tailed test p-value for z-score
    p_values = np.empty((num_total_snps), dtype=float)
    z_scores = z_score.iloc[:, 1]
    for i, z in enumerate(z_scores):
        p_value = scipy.stats.norm.sf(abs(z)) * 2
        p_values[i] = -1 * math.log10(p_value)

    causal_p_values = p_values[causal_index]
    non_causal_p_values = p_values[non_causal_index]

    plt.clf()
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    fig.suptitle("Inference results")
    ax1.scatter(causal_index, causal_p_values,
                c="red", label="True causal SNP")
    ax1.scatter(non_causal_index, non_causal_p_values,
                c="#2b70ad", label="Non-causal SNP", alpha=0.3)
    ax1.set_ylabel("-log10p")
    # ax1.set_xlabel("SNPs")
    ax1.grid(0.5)
    ax2.scatter(causal_index, causal_pips, c="red")
    ax2.scatter(non_causal_index, non_causal_pips, c="#2b70ad", alpha=0.3)
    ax2.set_ylabel("PIP")
    ax2.set_xlabel("SNPs")
    ax2.grid(0.5)
    fig.legend()
    fig.savefig("pip.png")
    plt.close()

    # PIP provided by professor
    real_pips = pip_real.iloc[:, 1].to_numpy()

    causal_pips = real_pips[causal_index]
    non_causal_pips = real_pips[non_causal_index]

    plt.clf()
    plt.scatter(causal_index, causal_pips,
                c="red", label="True causal SNP")
    plt.scatter(non_causal_index, non_causal_pips,
                c="#2b70ad", label="Non-causal SNP", alpha=0.3)
    plt.ylabel("PIP")
    plt.xlabel("SNPs")
    plt.title("PIP provided by professor")
    plt.grid(0.5)
    plt.legend()
    plt.savefig("ref_pip.png")
    plt.close()


if __name__ == "__main__":
    main()
