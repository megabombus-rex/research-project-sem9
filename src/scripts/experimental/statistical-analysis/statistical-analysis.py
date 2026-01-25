
import json
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

# 1. Funkcja pomocnicza do wczytywania
def load_json_results(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return {
        'name': data['model'].split('.')[-1].replace("'>", ""), 
        'f1': np.array(data['F1']),
        'auroc': np.array(data['auroc']),
        'map-macro': np.array(data['mAP-macro']),
        'map-micro': np.array(data['mAP-micro'])
    }


vit_mono_20 = load_json_results('./-ViT_multi_model_multilabel.ViTMonoMultilabelModel_10eps.json')
vit_tab_20 = load_json_results('./-ViT_multi_model_multilabel.ViTTabMultiModalMultilabelModel.json')
vit_text_20 = load_json_results('./-ViT_multi_model_multilabel.ViTTextMultiModalMultilabelModel.json')


def test(models: list, metrics_to_test: list):
    for metric_to_test in metrics_to_test:
        data_matrix = np.column_stack([m[metric_to_test] for m in models])
        
        ## Step 1: Friedman Test 

        stat, p_friedman = stats.friedmanchisquare(*data_matrix.T)
        print(f"--- FRIEDMAN TEST ({metric_to_test.upper()}) ---")
        print(f"Statistic: {stat:.4f}, p-value: {p_friedman:.4f}")
        ## KROK 2: Testy Post-hoc Wilcoxon + Holm (Priorytet 2)

        if p_friedman < 0.05:
            print("\nSignificant defferences detected. Begining comparing pairs...")
            
            pairs = [(0, 1), (0, 2), (1, 2)]
            p_values = []
            
            for i, j in pairs:
                # Test Wilcoxona dla sparowanych prób (te same foldy)
                _, p = stats.wilcoxon(data_matrix[:, i], data_matrix[:, j])
                p_values.append(p)
            
            # Korekta Holm (zapobiega błędom przy wielokrotnych porównaniach)
            reject, p_corrected, _, _ = multipletests(p_values, method='holm')
            
            results_table = []
            for idx, (i, j) in enumerate(pairs):
                results_table.append({
                    'Pair': f"{models[i]['name']} vs {models[j]['name']}",
                    'p-raw': p_values[idx],
                    'p-adj (Holm)': p_corrected[idx],
                    'Significant': reject[idx]
                })
            
            print(pd.DataFrame(results_table))
        else:
            print("\nNo grounds for rejecting H0. The models perform identically statistically. ")


print(vit_mono_20['f1'])
models = [vit_mono_20, vit_tab_20,vit_text_20]
metrics_to_test = ['f1', 'auroc', 'map-macro', 'map-micro']
test(models=models, metrics_to_test=metrics_to_test)


def cliffs_delta(x, y):
    lx, ly = len(x), len(y)
    count = 0
    for i in x:
        for j in y:
            if i > j: count += 1
            elif i < j: count -= 1
    d = count / (lx * ly)
    
    # Interpretacja siły efektu
    if abs(d) < 0.147: mag = "negligible"
    elif abs(d) < 0.33: mag = "small"
    elif abs(d) < 0.474: mag = "average"
    else: mag = "big"
    return d, mag

def analyze_models(model_data_dict, metric_name='auroc'):
    """
    model_data_dict: Słownik { 'Nazwa': lista_wyników_z_10_foldów }
    """
    df_results = pd.DataFrame(model_data_dict)
    model_names = list(model_data_dict.keys())
    
    print(f"=== DETAILED ANALYSIS FOR: {metric_name.upper()} ===\n")

    # KROK 1: Średnie i Odchylenia (Kierunek)
    print("1. DESCRIPTIVE STATISTICS ")
    summary = df_results.agg(['mean', 'std', 'median']).T
    print(summary)
    print("-" * 30)

    # KROK 2: Ranking Średni (Stabilność)
    ranks = df_results.rank(axis=1, ascending=False)
    mean_ranks = ranks.mean()
    print("\n2. Amounts ranking (1.0 = smallest, the smaller the better)")
    print(mean_ranks.sort_values())
    print("-" * 30)

    # KROK 3: Wielkość Efektu Cliff's Delta (Siła różnic)
    print("\n3. Effect size (CLIFF'S DELTA) for pairs")
    pairs = [(model_names[i], model_names[j]) 
             for i in range(len(model_names)) 
             for j in range(i+1, len(model_names))]
    
    for m1, m2 in pairs:
        d, mag = cliffs_delta(model_data_dict[m1], model_data_dict[m2])
        direction = m1 if d > 0 else m2
        print(f"{m1} vs {m2}: Delta = {d:.3f} | Force: {mag} (Better: {direction})")





dane = {
    'Mono': vit_mono_20['f1'],
    'Tabular': vit_tab_20['f1'],
    'Text': vit_text_20['f1']
}

analyze_models(dane, metric_name='f1')


dane = {
    'Mono': vit_mono_20['map-macro'],
    'Tabular': vit_tab_20['map-macro'],
    'Text': vit_text_20['map-macro']
}

analyze_models(dane, metric_name='map-macro')


dane = {
    'Mono': vit_mono_20['auroc'],
    'Tabular': vit_tab_20['auroc'],
    'Text': vit_text_20['auroc']
}

analyze_models(dane, metric_name='auroc')


dane = {
    'Mono': vit_mono_20['map-micro'],
    'Tabular': vit_tab_20['map-micro'],
    'Text': vit_text_20['map-micro']
}

analyze_models(dane, metric_name='map-micro')


# tests for full experiment


vit_mono_full = load_json_results('./-ViT_multi_model_multilabel.ViTMonoMultilabelModelv2_f5_r2_ep5_fulldataset.json')
vit_tab_full = load_json_results('./-ViT_multi_model_multilabel.ViTTabMultiModalMultilabelModelv2_f5_r2_ep5_fulldataset.json')
vit_text_full = load_json_results('./-ViT_multi_model_multilabel.ViTTextMultiModalMultilabelModelv2_f5_r2_ep5_fulldataset.json')





test(models=[vit_mono_full, vit_tab_full, vit_text_full], metrics_to_test=['f1', 'auroc', 'map-macro', 'map-micro'])


dane = {
    'Mono': vit_mono_full['f1'],
    'Tabular': vit_tab_full['f1'],
    'Text': vit_text_full['f1']
}

analyze_models(dane, metric_name='f1')


dane = {
    'Mono': vit_mono_full['map-macro'],
    'Tabular': vit_tab_full['map-macro'],
    'Text': vit_text_full['map-macro']
}

analyze_models(dane, metric_name='map-macro')


dane = {
    'Mono': vit_mono_full['auroc'],
    'Tabular': vit_tab_full['auroc'],
    'Text': vit_text_full['auroc']
}

analyze_models(dane, metric_name='auroc')


dane = {
    'Mono': vit_mono_full['map-micro'],
    'Tabular': vit_tab_full['map-micro'],
    'Text': vit_text_full['map-micro']
}

analyze_models(dane, metric_name='map-micro')


