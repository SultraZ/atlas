import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, numpy as np, json
from atlas.atlas_main import train_forward_solver
from atlas.validation import run_validation
from atlas.paths import RESULTS_DIR, FEM_REFERENCE_DIR

torch.manual_seed(42); np.random.seed(42); torch.cuda.manual_seed_all(42)
device = 'cuda'
model, trainer, sampler, phy = train_forward_solver(
    'cantilever', device, epochs=2000, print_every=500,
    n_interior=1000, n_boundary=200, n_interface=100,
    hidden_layers=5, neurons=128, warm_start_eb=True, prefit_epochs=6000,
    use_lbfgs=True, lbfgs_steps=15, lbfgs_max_iter_per_step=50,
    formulation='residual',   # the EB warm start is a residual-solver feature
)
os.makedirs(str(RESULTS_DIR), exist_ok=True)
torch.save(model.state_dict(), str(RESULTS_DIR / "_warm_cantilever_model.pt"))
res = run_validation(model, sampler, phy, benchmark="cantilever", device=device,
                     use_mock=False, data_dir=str(FEM_REFERENCE_DIR))
res["device"] = device
with open(str(RESULTS_DIR / "_validate_warm_cantilever.json"), "w") as f:
    json.dump(res, f, indent=2)
print("VALIDATION:")
for k, v in res.items():
    print(f"  {k}: {v}")