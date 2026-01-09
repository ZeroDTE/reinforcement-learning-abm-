import matplotlib.pyplot as plt
import numpy as np


def plot_learning_curves(scores, agent_names):
          num_agents = len(agent_names)
          fig, axes = plt.subplots(num_agents, 1, figsize=(12, 3*num_agents), sharex=True)

    if num_agents == 1:
                  axes = [axes]

    colors = ['red', 'green', 'blue', 'purple', 'orange']

    for i, ax in enumerate(axes):
                  ax.plot(scores[i], alpha=0.6, linewidth=1, color=colors[i])
                  ax.set_title(agent_names[i])
                  ax.set_ylabel('Score')
                  ax.grid(True, alpha=0.3)

        window = 50
        if len(scores[i]) > window:
                          smoothed = np.convolve(scores[i], np.ones(window)/window, mode='valid')
                          ax.plot(smoothed, color=colors[i], linewidth=2, label='smoothed')
                          ax.legend()

    axes[-1].set_xlabel('Episode')
    fig.suptitle('Learning Curves')
    plt.tight_layout()
    return fig


def plot_bars(results, names):
          fig, ax = plt.subplots(figsize=(10, 6))
          colors = ['red', 'green', 'blue', 'purple']
          scores = [results[n] for n in names]

    ax.bar(names, scores, color=colors[:len(names)], alpha=0.7)
    ax.set_ylabel('Average Score')
    ax.set_title('Agent Comparison')
    ax.grid(True, alpha=0.3, axis='y')

    for i, (name, score) in enumerate(zip(names, scores)):
                  ax.text(i, score + 1, f'{score:.1f}', ha='center', va='bottom')

    plt.tight_layout()
    return fig


def smooth(scores, window=50):
          if len(scores) < window:
                        return scores
                    return np.convolve(scores, np.ones(window)/window, mode='valid')


def stats(scores, name=""):
          scores = np.array(scores)
    print(f"Stats for {name}")
    print(f"Mean:   {np.mean(scores):.2f}")
    print(f"Max:    {np.max(scores):.2f}")
    print(f"Min:    {np.min(scores):.2f}")
    print(f"Std:    {np.std(scores):.2f}")
