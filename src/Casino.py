import numpy as np
import numpy.testing as npt
import seaborn as sns
import pylab as pb

## Global State

num_players = 1
K = 25
num_tables = 2*K
die_sides = 6
timesteps = K

# Dice distributions
pval_uniform = np.array([1/float(die_sides)] * die_sides)
pval_goodbias = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.5])
pval_badbias = np.array([0.5, 0.1, 0.1, 0.1, 0.1, 0.1])

# Ensure that dice distributions are stochastic
npt.assert_almost_equal(np.sum(pval_uniform), 1)
npt.assert_almost_equal(np.sum(pval_badbias), 1)
npt.assert_almost_equal(np.sum(pval_goodbias), 1)

#Initial state distribution
initial_state = np.array([1.0, 0])

# Probability of transition from table to table
transition_matrix = np.array([[0.25, 0.75],
                              [0.75, 0.25]])

#Probability of emitting observation in state i
emission_matrix = np.vstack([pval_uniform, pval_goodbias, pval_badbias])

# Assign the player a dice distribution
pval_player_bias = pval_uniform

def convolution(d1, d2):
    conv_dist = np.convolve(d1,d2)
    #Insert two zero's at index 0 and 1 to show a 0 probability for rolling a sum of 0 or 1.
    return np.insert(conv_dist, 0, [0,0])

def visit_tables(emission_matrix, transition_matrix, table_dist_index, tableprime_dist_index, player_dice_dist):
    """

    :rtype: list
    """
    current_table = [0] #0 for table, 1 for table_prime
    player_rolls = np.zeros(K)
    table_rolls = np.zeros(K)
    dice_sums = np.zeros(K)
    for i in range(K):
        # Transition to a new table, and fetch the appropriate dice distribution
        current_table = np.random.choice([table_dist_index,tableprime_dist_index], 1, True, transition_matrix[current_table[0],:])
        table_rolls[i] = dice_roll(emission_matrix[current_table[0],:])
        player_rolls[i] = dice_roll(emission_matrix[player_dice_dist,:])
        dice_sums[i] = table_rolls[i] + player_rolls[i]
    return dice_sums

def dice_roll(distribution):
    return np.random.choice([1,2,3,4,5,6], 1, True, distribution)

def fwd(init_state, observations, emission_matrix, transition_matrix):
    # An implementation of the forward algorithm presented in the HMM Tutorial by Stamp
    # The indices are flipped because I prefer to have the states as rows and the columns as time

    """
    :param init_state: Initial state distribution (is often pi in the literature)
    :param observations: Vector of observations
    :param emission_matrix:
    :param transition_matrix:
    :return:
    """
    c0 = 0
    num_obs = observations.shape[0]
    N = emission_matrix.shape[0]
    alpha = np.zeros((N,num_obs))
    alpha_normal = np.zeros((N,num_obs))

    for i in range(N):
        alpha[i][0] = init_state[i]*emission_matrix[i][observations[0]]
        c0 = c0 + alpha[i][0]

    #Scale the a_0(i)
    c0 = 1/c0
    for i in range(N):
        alpha_normal[i][0] = c0*alpha[i][0]

    #compute a_t(i)
    for t in range(1,num_obs):
        ct = 0
        for i in range(N):
            alpha[i][t] = 0
            for j in range(N):
                alpha[i][t] = alpha[i][t] + alpha[j][t-1]*transition_matrix[j][i]
            alpha[i][t] = alpha[i][t] * emission_matrix[i][observations[t]]
            ct = ct + alpha[i][t]

        #Scale a_t(i)
        ct = 1/ct
        for i in range(N):
            alpha_normal[i][t] = ct*alpha[i][t]

    return alpha, alpha_normal


def uniform_case():
# Visit the tables: tables and players use fair dice
    overall_observations = np.zeros((num_players, K))
    for player in range(num_players):
        overall_observations[player] = visit_tables(emission_matrix, transition_matrix, 0,0,0)
    sns.distplot(np.concatenate(overall_observations),bins=11, kde=False, rug=True)
    sns.plt.title("Casino: Dice Sums with Fair Dice")
    sns.plt.xlabel("Dice Sum")
    sns.plt.ylabel("# of Observations")
    pb.savefig("img/Casino_Uniform.png")
    sns.plt.show()
    return overall_observations


def biased_table_case():
    overall_observations = np.zeros((num_players, K))
    for player in range(num_players):
        overall_observations[player] = visit_tables(emission_matrix, transition_matrix, 0,1,0)
    sns.distplot(np.concatenate(overall_observations),bins=11, kde=False, rug=True)
    sns.plt.title("Casino: Dice Sums with Biased and Uniform Tables")
    sns.plt.xlabel("Dice Sum")
    sns.plt.ylabel("# of Observations")
    pb.savefig("img/Casino_Biased.png")
    sns.plt.show()
    return overall_observations


def dishonest_casino():
    overall_observations = np.zeros((num_players, K))
    for player in range(num_players):
        overall_observations[player] = visit_tables(emission_matrix, transition_matrix, 0,1,2)
    sns.distplot(np.concatenate(overall_observations),bins=11, kde=False, rug=True)
    sns.plt.title("Casino: Dice Sums with Biased Dice")
    sns.plt.xlabel("Dice Sum")
    sns.plt.ylabel("# of Observations")
    pb.savefig("img/Casino_Dishonest.png")
    sns.plt.show()
    return overall_observations

def draw_samples(probabilities):
    """

    :param probabilities:
    :return:
     :rtype: numpy array
    """
    primed_probs = normal_alphas[1,:]
    samples = np.zeros(primed_probs.shape[0])
    for i, prob in enumerate(primed_probs):
        samples[i] = np.random.binomial(1, prob)
    return samples




if __name__ == "__main__":

    ## Get our convolved dice distributions
    ## Generate an observation sequence
    ## Do the forward pass, and grab the prob of the final state

    pval_sums_uniform = convolution(pval_uniform, pval_uniform)
    pval_sums_biased = convolution(pval_uniform, pval_goodbias)
    sums_emission_matrix = np.vstack((pval_sums_uniform,pval_sums_biased))
    observations = np.ones((25,))*10
    alphas, normal_alphas = fwd(initial_state, observations, sums_emission_matrix,transition_matrix)
    samples = draw_samples(normal_alphas[1,:])
    time = np.linspace(0, samples.shape[0],samples.shape[0])

    pb.show()


    observations = visit_tables(emission_matrix,transition_matrix, 0, 1, 0)
    alphas, normal_alphas = fwd(initial_state, observations, sums_emission_matrix,transition_matrix)
    samples = draw_samples(normal_alphas[1,:])
    time = np.linspace(0, samples.shape[0],samples.shape[0])

    pb.plot(time,samples)
    pb.plt.ylim(-0.5,1.5)
    pb.plt.title("Sampled State Sequence at Casino")
    pb.ylabel("State [0] = unprimed")
    pb.xlabel("Time")
    pb.plt.savefig("img/Sampling_StateSequence_gen_obs.png")
    pb.plot(time,samples)
    pb.plt.ylim(-0.5,1.5)
    pb.plt.title("Sampled State Sequence at Casino")
    pb.ylabel("State [0] = unprimed")
    pb.xlabel("Time")
    pb.plt.savefig("img/Sampling_StateSequence_all_obs_10.png")
    pb.show()


