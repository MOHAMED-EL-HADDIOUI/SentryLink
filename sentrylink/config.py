"""Global cryptographic and numeric parameters for SentryLink."""

# Prime field for additive secret sharing (Mersenne prime 2^127 - 1).
FIELD_P = (1 << 127) - 1

# Fixed-point scale for quantizing float model updates / stats before
# integer masking or secret sharing.
QUANT_SCALE = 1_000_000

# Pairwise secure-aggregation masks are drawn from [-MASK_BOUND, MASK_BOUND]
# so that int64 sums stay far below 2^63 for realistic roster sizes.
MASK_BOUND = 1 << 40

# L2 clip bound applied to every federated model update before masking/DP.
UPDATE_CLIP = 1.0

# Default Gaussian-DP parameters for a single protected release.
DEFAULT_EPSILON = 8.0
DEFAULT_DELTA = 1e-5

# Governance: a result is only released when at least this many distinct
# organizations contributed (k-anonymity style cohort floor).
MIN_PARTICIPANTS = 3

# Per-organization L2 contribution radius for released aggregate vectors
# (histogram buckets, etc.). One org's vector is projected onto this ball
# before sharing, so add/remove sensitivity of the joint release = radius.
MAX_ORG_CONTRIB = 100.0

ROUNDS_DEFAULT_CLIENT_EPOCHS = 2
ROUNDS_DEFAULT_LR = 0.1
