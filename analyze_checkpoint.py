import pickle
import numpy as np

data = pickle.load(open('ml_checkpoint.pkl', 'rb'))
print('=== ML CHECKPOINT ANALYSIS ===')
print(f'Total assets: {len(data["assets"])}')
print()

# Check structure
print('=== DATA STRUCTURE ===')
first_asset = data['assets'][0]
first_model = data['models'][first_asset]
print(f'Model type: {type(first_model)}')
if isinstance(first_model, dict):
    print(f'Model keys: {first_model.keys()}')
    for k, v in first_model.items():
        if hasattr(v, 'shape'):
            print(f'  {k}: array shape {v.shape}')
        elif isinstance(v, list):
            print(f'  {k}: list len {len(v)}')
        else:
            print(f'  {k}: {v}')
else:
    print(f'Model attributes: {[a for a in dir(first_model) if not a.startswith("_")]}')
    
print()
print('=== SAMPLE MODELS ===')
for i, asset in enumerate(data['assets'][:5]):
    m = data['models'][asset]
    if isinstance(m, dict):
        print(f'{asset}: n_updates={m.get("n_updates", "N/A")}, theta_mean={np.mean(m.get("theta", [0])):.6f}')
    else:
        print(f'{asset}: n_updates={m.n_updates}, theta_mean={m.theta.mean():.6f}')

