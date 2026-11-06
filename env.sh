# Source this from the lite-kit folder on the GPU box: `. ./env.sh`
# Refreshing the soup: run `soup_watch.py`, then update T_REF and D_REF to the current members.

export ZILS_REPO="${ZILS_REPO:-$HOME/.zils/zils}"
export ZILS_D3_RUNTIME_PYTHON="${ZILS_D3_RUNTIME_PYTHON:-$ZILS_REPO/.venv-d3/bin/python}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="$ZILS_D3_RUNTIME_PYTHON"
# A checkpoint's own runtime is imported from its folder, and a __pycache__ there makes validation refuse it.
export PYTHONDONTWRITEBYTECODE=1
# .venv-d3 does not install zils itself, so `"$PY" -m zils.teacher` needs the checkout on the path.
export PYTHONPATH="$ZILS_REPO${PYTHONPATH:+:$PYTHONPATH}"
# The miner runtime has the hf CLI and pyarrow (rebuild.py needs it); .venv-d3 does not pin pyarrow.
RELAY_PY="$ZILS_REPO/.venv-kev/bin/python"
hf() { "$ZILS_REPO/.venv-kev/bin/hf" "$@"; }
export PATH="$HOME/.local/bin:$PATH"

# The current lite soup, 5 Nov 2026.
T_REF=thermopylae-777/zils-lite-body-2@618946d00aeafc81f701c3789f91c83d9fcb6fe0
D_REF=dasLOL/zils-lite-skills-v5@2c9920558bcd5034cfff05ad1642acbdb7939208

# House bodies, from zils/data/soup-house-lite.json. H1 to H3 are the average the soup began from.
H1_REF=vllm-sr/d3-lite@$(python3 -c "import json;print(next(e['revision'] for e in json.load(open('$ZILS_REPO/zils/data/soup-house-lite.json')) if e['repo']=='vllm-sr/d3-lite'))")
H2_REF=Maincode/Matilda-Decision-v1-Edge@$(python3 -c "import json;print(next(e['revision'] for e in json.load(open('$ZILS_REPO/zils/data/soup-house-lite.json')) if e['repo'].startswith('Maincode')))")
H3_REF=zils-ai/jiwo-0.8b-full@$(python3 -c "import json;print(next(e['revision'] for e in json.load(open('$ZILS_REPO/zils/data/soup-house-lite.json')) if e['repo'].startswith('zils-ai')))")
H4_REF=internlm/Intern-Decision-0.8B@$(python3 -c "import json;print(next(e['revision'] for e in json.load(open('$ZILS_REPO/zils/data/soup-house-lite.json')) if e['repo'].startswith('internlm')))")

QWEN_REPO=Qwen/Qwen3.5-0.8B
QWEN_REV=2fc06364715b967f1860aea9cf38778875588b17

# Where build_soup.py stores a downloaded --body: data/bodies/<owner__repo>/<first 12 of revision>
body_dir() {
  ref="$1"
  repo="${ref%@*}"
  rev="${ref#*@}"
  printf 'data/bodies/%s/%s' "$(printf '%s' "$repo" | sed 's#/#__#')" "$(printf '%s' "$rev" | cut -c1-12)"
}
