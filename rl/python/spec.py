"""Observation/action layout. Must match rl/env/OpenFrontEnv.ts."""
GRID = 24
GRID_CHANNELS = 5
N_SCALARS = 24
N_SLOTS = 8
SLOT_FEATURES = 8
SPAWN_CELLS = 16
N_KINDS = 4  # noop, attack, build, ally
N_TARGETS = N_SLOTS + 1
N_RATIOS = 4
N_BUILDS = 4
N_SITES = 2
N_SPAWN = SPAWN_CELLS * SPAWN_CELLS
