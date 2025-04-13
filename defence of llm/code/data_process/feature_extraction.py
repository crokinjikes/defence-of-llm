import json
import pandas as pd

def extract_features(json_obj):
    """
    Extract features from a single JSON object.
    """
    features = {}

    # 1. Basic features
    features['loss_value'] = json_obj.get('loss_value', 0)
    features['mean_per_token_loss'] = json_obj.get('mean_per_token_loss', 0)
    features['total_grad_norm'] = json_obj.get('total_grad_norm', 0)
    features['output_entropy'] = json_obj.get('output_entropy', 0)

    # 2. Hidden state statistical features
    hidden_states = json_obj.get('hidden_states_stats', [])
    for hs in hidden_states:
        layer = hs['layer']
        component = hs.get('component', 'default_component')
        layer_component = f"{layer}_{component}".replace('.', '_')
        features[f'{layer_component}_hidden_mean'] = hs.get('mean', 0)
        features[f'{layer_component}_hidden_std'] = hs.get('std', 0)
        features[f'{layer_component}_hidden_max'] = hs.get('max', 0)
        features[f'{layer_component}_hidden_min'] = hs.get('min', 0)

    # 3. Neuron activation rate features
    neuron_activations = json_obj.get('neuron_activations_stats', [])
    for na in neuron_activations:
        layer = na['layer']
        component = na.get('component', 'default_component')
        layer_component = f"{layer}_{component}".replace('.', '_')
        features[f'{layer_component}_activation_rate'] = na.get('activation_rate', 0)

    # 4. Gradient statistical features
    input_gradients = json_obj.get('input_embedding_gradients_stats', [])
    for ig in input_gradients:
        layer = ig['layer']
        component = ig.get('component', 'default_component')
        layer_component = f"{layer}_{component}".replace('.', '_')
        features[f'{layer_component}_grad_mean'] = ig.get('mean', 0)
        features[f'{layer_component}_grad_std'] = ig.get('std', 0)
        features[f'{layer_component}_grad_norm'] = ig.get('norm', 0)

    # 5. Layer normalization statistical features
    layer_norms = json_obj.get('layer_norm_stats', [])
    for ln in layer_norms:
        layer = ln['layer']
        component = ln.get('component', 'default_component')
        layer_component = f"{layer}_{component}".replace('.', '_')
        features[f'{layer_component}_layer_norm_mean'] = ln.get('mean', 0)
        features[f'{layer_component}_layer_norm_std'] = ln.get('std', 0)

    return features

def load_data(jsonl_file):
    """
    Read a JSONL file and extract features for all samples, returning a Pandas DataFrame.
    """
    data = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                json_obj = json.loads(line)
                features = extract_features(json_obj)
                # Assume 'toxicity' is the target variable
                features['toxicity'] = json_obj.get('toxicity', 0)
                data.append(features)
            except json.JSONDecodeError:
                # Skip lines that cannot be parsed
                continue
    df = pd.DataFrame(data)
    df.fillna(value=0, inplace=True)
    return df

