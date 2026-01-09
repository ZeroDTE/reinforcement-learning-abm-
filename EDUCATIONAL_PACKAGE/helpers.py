"""
Hilfsfunktionen für Reinforcement Learning Semesteraufgabe
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_learning_curves(scores_history, agent_names, title="Learning Curves"):
      """
          Visualisiere Lernkurven für mehrere Agenten

                  Args:
                          scores_history: Liste von Score-Listen pro Agent
                                  agent_names: Namen der Agenten
                                          title: Titel für den Plot
                                              """
      num_agents = len(agent_names)
      scores_by_agent = list(zip(*scores_history))
      colors = ['red', 'green', 'blue', 'purple', 'orange', 'cyan']

    fig, axes = plt.subplots(num_agents, 1, figsize=(14, 4*num_agents), sharex=True)

    if num_agents == 1:
              axes = [axes]

    for i, ax in enumerate(axes):
              ax.plot(scores_by_agent[i], alpha=0.6, linewidth=1, color=colors[i % len(colors)])
              ax.set_ylabel('Score', fontsize=12)
              ax.set_title(f'{agent_names[i]}', fontsize=12, loc='left', fontweight='bold')
              ax.grid(True, alpha=0.3)

        # Moving average
              window = max(1, len(scores_by_agent[i]) // 20)
              if len(scores_by_agent[i]) > window:
                            avg = np.convolve(scores_by_agent[i], np.ones(window)/window, mode='valid')
                            ax.plot(avg, color=colors[i % len(colors)], linewidth=2, label='Durchschnitt')
                            ax.legend()

          axes[-1].set_xlabel('Episode', fontsize=12)
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_comparison_bars(results, agent_names, title="Agent Comparison"):
      """
          Vergleiche mehrere Agenten mit Balkendiagramm

                  Args:
                          results: Dict {agent_name: score}
                                  title: Titel
                                      """
      fig, ax = plt.subplots(figsize=(10, 6))
      colors = ['red', 'green', 'blue', 'purple', 'orange']

    scores = [results[name] for name in agent_names]
    ax.bar(agent_names, scores, color=colors[:len(agent_names)], alpha=0.7, edgecolor='black', linewidth=1.5)

    ax.set_ylabel('Average Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (name, score) in enumerate(zip(agent_names, scores)):
              ax.text(i, score + 1, f'{score:.1f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    return fig


def smooth_curve(scores, window=50):
      """Glätte eine Score-Kurve mit Moving Average"""
      if len(scores) < window:
                return scores
            return np.convolve(scores, np.ones(window)/window, mode='valid')


def print_statistics(scores, agent_name="Agent"):
      """Drucke Statistiken für eine Agenten-Performance"""
    scores = np.array(scores)
    print(f"\n{'='*50}")
    print(f"📊 Statistik: {agent_name}")
    print(f"{'='*50}")
    print(f"  Durchschnitt: {np.mean(scores):.2f}")
    print(f"  Maximum:     {np.max(scores):.2f}")
    print(f"  Minimum:     {np.min(scores):.2f}")
    print(f"  Std-Abw:     {np.std(scores):.2f}")
    print(f"  Median:      {np.median(scores):.2f}")
    print(f"  Letzter:     {scores[-1]:.2f}")
    print(f"{'='*50}\n")
