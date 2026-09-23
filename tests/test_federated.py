import numpy as np

from sentrylink.federated.client import FederatedClient
from sentrylink.federated.model import LogisticModel, evaluate_sufficient_stats
from sentrylink.federated.server import FederatedServer
from sentrylink.usecases import make_retail_cohort


def _clients(orgs, ids):
    return [FederatedClient(org_id=oid, x=o.x, y=o.y) for oid, o in zip(ids, orgs)]


def test_local_training_reduces_own_loss():
    orgs = make_retail_cohort(n_orgs=1, n_per_org=300, seed=1)
    c = FederatedClient(org_id="a", x=orgs[0].x, y=orgs[0].y)
    model = LogisticModel.zeros(4)
    _, loss_before, _ = evaluate_sufficient_stats(model, c.x, c.y)
    delta = c.local_train(model, epochs=3)
    m2 = LogisticModel.from_flat(4, model.flat + delta)
    _, loss_after, _ = evaluate_sufficient_stats(m2, c.x, c.y)
    assert loss_after < loss_before


def test_federated_round_learns():
    orgs = make_retail_cohort(n_orgs=4, n_per_org=350, seed=7)
    clients = _clients(orgs, [o.org_id for o in orgs])
    server = FederatedServer(dim=4)
    # initial accuracy
    stats0 = server._secure_eval(clients, server.model)
    for _ in range(6):
        result = server.run_round(clients, epochs=2, apply_dp=True, epsilon=8.0)
    stats1 = result.eval_stats
    assert stats1["accuracy"] > stats0["accuracy"]
    assert stats1["accuracy"] > 0.6


def test_federated_round_with_dropout():
    orgs = make_retail_cohort(n_orgs=4, n_per_org=200, seed=7)
    clients = _clients(orgs, [o.org_id for o in orgs])
    server = FederatedServer(dim=4)
    dropped = clients[-1].org_id
    r = server.run_round(clients, epochs=1, drop=[dropped], apply_dp=False)
    assert r.dropped == [dropped]
    assert dropped not in r.participants
    assert len(r.participants) == 3


def test_dp_noise_actually_perturbs_aggregate():
    orgs = make_retail_cohort(n_orgs=3, n_per_org=150, seed=4)
    clients = _clients(orgs, [o.org_id for o in orgs])
    s1 = FederatedServer(dim=4)
    s2 = FederatedServer(dim=4)
    rng1 = np.random.default_rng(1)
    rng2 = np.random.default_rng(2)
    r1 = s1.run_round(clients, epochs=1, apply_dp=True, epsilon=0.5, rng=rng1)
    r2 = s2.run_round(clients, epochs=1, apply_dp=True, epsilon=0.5, rng=rng2)
    assert not np.allclose(r1.aggregate_delta, r2.aggregate_delta)


def test_eval_stats_are_pooled_not_per_client():
    orgs = make_retail_cohort(n_orgs=3, n_per_org=100, seed=2)
    clients = _clients(orgs, [o.org_id for o in orgs])
    stats = FederatedServer._secure_eval(clients, LogisticModel.zeros(4))
    assert stats["n"] == 300
    assert "per_client" not in stats
