### `Requirements`

- Python 3.11+ on Windows 10, Windows 11 or Linux (X11/Wayland).

    python -m pip install -r requirements.txt
    python launcher.py

- `launcher.py` is the entry point, it selects a verified update if one is staged and falls back to the bundled version otherwise.
- `python app.py` runs the bundled version only and never applies updates.

<img width="1438" height="831" alt="front" src="https://github.com/user-attachments/assets/b632802d-be53-4d13-b5fc-23a730498c0b" />

### `What the agents do`

      | Mechanism | Technique |
      |---|---|
      | Movement | Lévy-flight exploration around a personal home anchor, sweep-and-prune spatial index for separation (personal space 34 units), crowd avoidance, resource ecology scaled to population size |
      | Dialogue strategy | Per-agent discounted Thompson sampling (Beta posteriors) over 8 speech acts in 16 contexts; listeners imitate successful acts (cultural transmission) |
      | Vocabulary | Naming game with lateral inhibition: agents invent words, adopt each other's terms and converge into local dialects |
      | Phrasing | Utterances are filled from real state (knowledge levels, resource bearings, opinion distance); phrasing becomes more elaborate as experience grows |
      | Awareness of the observer | Each agent runs a 3-state hidden Markov filter (ignored / glanced at / in focus) over a noisy signal of your attention. 
      | visible area, zoom, foveal weighting toward the view centre, cursor proximity weighted by dwell (a resting cursor counts more than a sweeping one) and selection. A hysteresis band (0.65 on / 0.35 off) keeps false alarms around 0.3% of agent-cycles when nobody watches |
      | Metacognition and emotion | Certainty is the normalised entropy of the posterior. Arousal is driven by novelty and surprise and habituates under prolonged watching, so agents keep knowing they are watched but stop reacting as strongly. Bold agents perform and comment; reserved agents whisper and flee the cursor in proportion to arousal |
      | Social awareness | An alert is treated as soft evidence: the listener fuses the speaker's posterior into its own, weighted by trust, so awareness spreads through conversation |
      | Learning across runs | At the end of each run the population's learned speech-act statistics (excluding anything they inherited) and the leading words per topic are merged, with decay, into an encrypted local knowledge record. New runs start from it: strategy priors capped at 20 pseudo-observations and a 60% chance of inheriting each common word. On three unseen seeds the first 120 cycles reached a mean dialogue reward of about 0.49 vs 0.32 cold, and lexical understanding of about 0.70 vs 0.40 |

### `Dialogue knowledge and community packs`

- Use the **Conhecimento** menu to turn inheritance on or off, export your pack, import community packs or remove imported packs.

- A pack holds only aggregated statistics. It never contains transcripts, memories or anything about the user, and it is never uploaded automatically.
- Imports are strictly validated. Unsigned packs count for at most 200 observations per cell, and packs signed by a key listed in `community_keys.txt` for at most 1000.
- Contributions are combined with a weighted median, and a word is adopted only when a majority of packs agree on it.
- `community/` is a template for a public GitHub repository that collects packs through pull requests, validated in CI by `knowledge_tool.py validate`.
- The median only protects you while honest packs are the majority, so keep the trusted key list short and review contributions.

- The knowledge table is added without changing the store's schema version, so rolling back to 9.0 still opens the same database.

### `Tests`

      python -m pip install -r requirements-dev.txt
      python -m pytest
      HYPOTHESIS_PROFILE=ci python -m pytest
