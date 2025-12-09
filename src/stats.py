import matplotlib.pyplot as plt
import statistics
import numpy as np
import seaborn as sns
import matplotlib as mpl
from tqdm import tqdm
from matplotlib.pyplot import figure
from torch_geometric.nn import summary

mpl.rcParams['figure.dpi'] = 200

def train_stats(agent: object, N=3, steps=10_000, savefig=True):
  """Used to gather training statistics for the agent"""

  mean, std, total = [], [], []
  for _ in range(N):
    rewards, links = agent.train(episodes = steps, plot=False, save_model=False)
    mean.append(rewards)
    std.append(statistics.stdev(rewards))
    total.append(links)

  avg_reward = np.mean(mean, axis=0)
  avg_std = statistics.mean(std)
  avg_links = np.mean(total)

  x = np.arange(len(avg_reward))
  plt.plot(x, avg_reward, 'tab:blue', ls='-', label=f'Mean (l: {avg_links:.0f} r: {avg_reward[-1]:.0f})')
  plt.fill_between(x, avg_reward-avg_std, avg_reward+avg_std, color='blue', alpha=0.2, label=f'Std:{avg_std:.1f}')
  plt.title(f'Reward over {N} runs of {steps} steps (GNN)')
  plt.xlabel('Step')
  plt.ylabel(f'Training reward for $(n, p_E, p_S)$= {agent.n, agent.p_entangle, agent.p_swap}')
  plt.legend()
  plt.yscale("symlog")
  plt.savefig('assets/train_stats.png') if savefig else None
  plt.show()


def test_stats(agent, 
               experiments=3, 
               n_test=4, 
               p_entangle=0.8, 
               p_swap=0.8, 
               tau=1000, 
               cutoff = 10_000,
               rounds=400, 
               quiet=False, 
               savefig=True):
  
  """Used to gather validation statistics for the agent"""

  t_err, s_err, r_err = [], [], []
  t_mean, s_mean, r_mean = [], [], []

  params = {
    'n_test': n_test,
    'p_entangle': p_entangle,
    'p_swap': p_swap,
    'tau': tau,
    'cutoff': cutoff,
    'max_steps': rounds
  }

  for _ in tqdm(range(experiments)):
    _, _, lt = agent.test(**params, kind='trained',);
    _, _, ls = agent.test(**params, kind='swap_asap');
    _, _, lr = agent.test(**params, kind='random');

    t_mean.append(lt)
    s_mean.append(ls)
    r_mean.append(lr)

    t_err.append(statistics.stdev(lt))
    s_err.append(statistics.stdev(ls))
    r_err.append(statistics.stdev(lr))

  lt = np.mean(t_mean, axis=0)
  ls = np.mean(s_mean, axis=0)
  lr = np.mean(r_mean, axis=0)

  t_err = statistics.mean(t_err)
  s_err = statistics.mean(s_err)
  r_err = statistics.mean(r_err)

  error_t = np.random.normal(0, t_err, size=len(lt))
  error_s = np.random.normal(0, s_err, size=len(ls))
  error_r = np.random.normal(0, r_err, size=len(ls))

  if quiet == True:
      return lt[-1], ls[-1], lr[-1], t_err, s_err, r_err

  x = np.arange(len(lt))

  plt.plot(x, lt, 'tab:blue', ls='-', label=f'Trained agent (rate: {lt[-1]:.3f})')
  plt.fill_between(x, lt-error_t, lt+error_t, color='blue', alpha=0.2)

  plt.plot(x, ls, 'tab:green', ls='-', label=f'Swap asap (rate: {ls[-1]:.3f})')
  plt.fill_between(x, ls-error_s, ls+error_s, color='green', alpha=0.2)

  plt.plot(x, lr, 'tab:grey', ls='-', label=f'Random (rate: {lr[-1]:.3f})')
  plt.fill_between(x, lr-error_r, lr+error_r, color='grey', alpha=0.2)

  plt.plot()
  plt.title(f'Average performance over {experiments} runs')
  plt.xlabel('Round')
  plt.ylabel(f'Link rate for $(n, p_E, p_S, τ)$= {n_test, p_entangle, p_swap, tau}')
  plt.legend()
  plt.savefig('assets/test_stats.png') if savefig else None
  plt.close()

   #--- Plot performance stats ---
  means = [np.mean(lt), np.mean(ls), np.mean(lr)]
  mean_errors = [np.mean(t_err), np.mean(s_err), np.mean(r_err)]

  labels = ["Agent", "Swap ASAP", "Random"]
  plt.figure(figsize=(8, 6))
  plt.bar(
      x=labels,
      height=means,
      yerr=mean_errors,
      capsize=10,  # Cap width
      color=(0, 0, 0, 0), # Transparent fill
      edgecolor=".5",     # Gray outline
      linewidth=2.5,      # Thick outline
      error_kw={
          "ecolor": ".5",      # Color of error bars
          "elinewidth": 2.5,   # Width of error bar lines
          "markeredgewidth": 2.5 # Thickness of the caps
          })

  plt.ylabel("Mean Performance")
  plt.title("Performance by Strategy")
  sns.despine()
  plt.savefig('assets/test_comp.png') if savefig else None
  plt.close()



def n_scaling_test(experiments, 
                   agent, 
                   N_range:range, 
                   p_e, 
                   p_s, 
                   tau, 
                   cutoff,
                   savefig=False):
  """Used to test the agents relative scaling"""

  n_test=[n for n in N_range]
  lt_list, ls_list = [], []
  t_err_list, s_err_list = [], []

  for n in n_test:
    lt, ls, _, t_err, s_err, _ = test_stats(agent, 
                                            experiments=experiments, 
                                            n_test = n, 
                                            p_entangle=p_e, 
                                            p_swap=p_s, 
                                            cutoff=cutoff,
                                            tau=tau, 
                                            quiet=True)

    lt_list.append(lt)
    ls_list.append(ls)

    t_err_list.append(t_err)
    s_err_list.append(s_err)

    lt = np.array(lt_list)
    ls = np.array(ls_list)
    t_err = np.array(t_err_list)
    s_err = np.array(s_err_list)

    performance_array = (lt - ls) / (ls +1e-5)
    errors = np.sqrt((t_err / (ls +1e-5))**2 + (lt*np.log(ls)*s_err)**2) #correctly propagate the error

  figure(figsize=(12, 4), dpi=100)
  plt.plot([n_test[0],n_test[-1]], [0,0], color='g')
  plt.fill_between(n_test, -s_err, s_err, color='green', alpha=0.1)

  plt.plot(n_test, performance_array, 'gD', color='b', label=f'Relative performance')
  plt.fill_between(n_test, performance_array-errors, performance_array+errors, color='blue', alpha=0.1)
  plt.legend()
  plt.ylabel(f'$(p_e, p_s, τ)=(0.5, 0.5, 100)$')
  plt.title(f'Relative performance scaling')
  plt.xlabel(f'$n$')
  plt.xticks(n_test)
  plt.savefig('assets/n_scaling.png') if savefig else None
  plt.show()

