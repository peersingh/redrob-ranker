"""Generates redrob_sandbox.ipynb — run this once then commit the .ipynb"""
import json

nb = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {"provenance": []},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"}
    },
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Redrob Hackathon — Candidate Ranking Sandbox\n",
                "**team_solo** | Lakavath Peer Singh\n\n",
                "This notebook clones the ranker repo, installs dependencies, and runs `rank.py` on a small sample.\n",
                "It demonstrates end-to-end reproducibility within the 5-minute CPU compute budget.\n\n",
                "**Runtime:** ~2 minutes on CPU | **Network during ranking:** 0 calls"
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "# Step 1: Clone repo and install dependencies\n",
                "!git clone https://github.com/peersingh/redrob-ranker.git\n",
                "%cd redrob-ranker\n",
                "!pip install -q -r requirements.txt"
            ],
            "outputs": [],
            "execution_count": None
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "# Step 2: Prepare sample candidates as JSONL\n",
                "# sample_candidates.json contains 10 real candidates from the pool\n",
                "import json\n",
                "\n",
                "with open('data/sample_candidates.json') as f:\n",
                "    samples = json.load(f)\n",
                "\n",
                "# Write as JSONL for rank.py\n",
                "with open('data/candidates.jsonl', 'w') as f:\n",
                "    for c in samples:\n",
                "        f.write(json.dumps(c) + '\\n')\n",
                "\n",
                "print(f'Prepared {len(samples)} sample candidates as JSONL')"
            ],
            "outputs": [],
            "execution_count": None
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "# Step 3: Run rank.py\n",
                "# Uses pre-computed caches (rule_scores.parquet, llm_judgments.jsonl, etc.)\n",
                "# rank.py is CPU-only, zero network calls, completes in < 2 minutes\n",
                "import time\n",
                "t0 = time.time()\n",
                "!python rank.py --candidates ./data/candidates.jsonl --out ./team_solo.csv\n",
                "print(f'\\nCompleted in {time.time()-t0:.1f}s')"
            ],
            "outputs": [],
            "execution_count": None
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "# Step 4: Validate the submission\n",
                "!python data/validate_submission.py team_solo.csv"
            ],
            "outputs": [],
            "execution_count": None
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "# Step 5: Show top-10 results\n",
                "import pandas as pd\n",
                "df = pd.read_csv('team_solo.csv')\n",
                "print('=== TOP 10 RANKED CANDIDATES ===')\n",
                "print(df.head(10)[['rank','candidate_id','score','reasoning']].to_string(index=False))"
            ],
            "outputs": [],
            "execution_count": None
        },
        {
            "cell_type": "code",
            "metadata": {},
            "source": [
                "# Step 6: Download the CSV\n",
                "from google.colab import files\n",
                "files.download('team_solo.csv')"
            ],
            "outputs": [],
            "execution_count": None
        }
    ]
}

with open("redrob_sandbox.ipynb", "w") as f:
    json.dump(nb, f, indent=2)

print("Generated: redrob_sandbox.ipynb")
