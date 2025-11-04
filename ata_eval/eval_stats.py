import numpy as np
import csv
import os 
from os.path import join as pjoin
from utils.parser_util import ata_evaluation_stats_parser

def flatten_data(y):
    out = {}

    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '_')
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + '_')
                i += 1
        else:
            out[name[:-1]] = x

    flatten(y)
    return out


args = ata_evaluation_stats_parser()
with open(args.benchmark_path, 'r') as fr:
    evaluated_characters = [c.strip() for c in fr.readlines()]
assert 'bvh' not in args.eval_mode, 'bvh mode is not supported any longer!'

subset_type = "all"
if 'names' in args.benchmark_path:
    #models_path = ["save/all_model_l_simple_geodesic_trans.dec_dataset_truebones_bs_8_latentdim_128", "save/all_model_l_simple_geodesic_trans.dec_dataset_truebones_bs_16_latentdim_128", "save/all_model_l_simple_trans.dec_dataset_truebones_bs_8_latentdim_128"]
    models_path = ["/a/home/cc/students/cs/sigalraab/python/multi-skeleton-mdm-delete_after_250331/save/ablation_dist_cond_only_trans.dec_dataset_truebones_bs_16_latentdim_128", "/a/home/cc/students/cs/sigalraab/python/multi-skeleton-mdm-delete_after_250331/save/ablation_no_grpe_trans.dec_dataset_truebones_bs_16_latentdim_128", "/a/home/cc/students/cs/sigalraab/python/multi-skeleton-mdm-delete_after_250331/save/ablation_no_tpos_concat_trans.dec_dataset_truebones_bs_16_latentdim_128", "/a/home/cc/students/cs/sigalraab/python/multi-skeleton-mdm-delete_after_250331/save/ablation_relations_cond_only_trans.dec_dataset_truebones_bs_16_latentdim_128", "save/all_model_l_simple_geodesic_skip_t5_trans.dec_dataset_truebones_bs_16_latentdim_128"]
    subset_type = "ablation"
elif 'dinosaurs' in args.benchmark_path:    
    subset_type="dinosaurs"
    models_path = ["save/dinosaurs_model_geodesic_and_ric_trans.dec_dataset_truebones_bs_8_latentdim_128", "save/dinosaurs_model_geodesic_trans.dec_dataset_truebones_bs_8_latentdim_128", "save/dinosaurs_model_l_simple_trans.dec_dataset_truebones_bs_8_latentdim_128"]
elif 'mammals' in args.benchmark_path:    
    subset_type="mammals"
    models_path = ["save/mammals_model_l_simple_trans.dec_dataset_truebones_bs_8_latentdim_128", "save/mammals_model_geodesic_trans.dec_dataset_truebones_bs_8_latentdim_128", "save/mammals_model_geodesic_and_ric_trans.dec_dataset_truebones_bs_8_latentdim_128"]
elif 'insects' in args.benchmark_path:  
    subset_type="insects"  
    models_path = ["save/insects_model_geodesic_trans.dec_dataset_truebones_bs_8_latentdim_128", "save/insects_model_geodesic_and_ric_trans.dec_dataset_truebones_bs_8_latentdim_128"]
elif 'flying' in args.benchmark_path:    
    subset_type="flying"
    models_path = ["save/flying_model_geodesic_and_ric_trans.dec_dataset_truebones_bs_8_latentdim_128", "save/flying_model_geodesic_trans.dec_dataset_truebones_bs_8_latentdim_128", "save/flying_model_l_simple_trans.dec_dataset_truebones_bs_8_latentdim_128", "/home/dcor/inbargat1/multi-skeleton-mdm/save/flying_objects_only_trans.dec_dataset_truebones_bs_8_latentdim_128"]

else: 
    raise NotImplementedError("benchmark file not supported")

if args.eval_mode == 'npy_rot':
    eval_file_name = "eval_npy_mode_npy_rot.npy"
elif args.eval_mode == 'npy_loc':
    eval_file_name = "eval_npy_mode_npy_loc.npy"
else: # 'npy_relative_loc'
    eval_file_name = "eval_npy_mode_npy_loc_rel_y.npy"

    
    
fields = ["model", "iteration", "coverage_mean", "coverage_std", "global_diversity_mean", "global_diversity_std", "local_diversity_mean", 
          "local_diversity_std", "inter_diversity_dist_mean", "inter_diversity_dist_std", "intra_diversity_dist_mean",
          "intra_diversity_dist_std", "gt_intra_diversity_dist_mean", "gt_intra_diversity_dist_std", "intra_div_gt_diff_mean", "intra_div_gt_diff_std"]

eval_files = list()
for model_path in models_path:
    model_eval_files = [pjoin(model_path, d) for d in os.listdir(model_path) if d.startswith("eval_") and os.path.exists(pjoin(model_path, d, 'eval_npy_mode_npy_loc_rel_y.npy'))]
    eval_files += model_eval_files
    
eval_results_for_csv = list()
eval_results_full_for_csv = list()
for eval_file in eval_files:
    model_name = eval_file.split('/')[-1].split('_')[2:-2]
    model_name = "_".join(model_name)
    iteration = int(eval_file.split('_')[-2])
    eval_results = np.load(pjoin(eval_file, eval_file_name), allow_pickle=True).item()
    if len(eval_results.keys()) != len(evaluated_characters) + 1:
        print(f"{eval_file} is incomplete")
        continue
    model_dict = {"model": model_name, "iteration": iteration}
    flatten_results = flatten_data(eval_results["FINAL"])
    flatten_results.update(model_dict)
    eval_results_for_csv += [flatten_results]
    for topo in eval_results.keys():
        topo_res = eval_results[topo].copy()
        topo_res["topology"] = topo
        flatten_topo_res = flatten_data(topo_res)
        flatten_topo_res.update(model_dict)
        eval_results_full_for_csv += [flatten_topo_res]
          
with open(pjoin("ata_eval", f"{subset_type}_eval_results.csv"),"w",newline="") as f:  
    cw = csv.DictWriter(f,fields,delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    cw.writeheader()
    cw.writerows(eval_results_for_csv)
    
with open(pjoin("ata_eval", f"{subset_type}_eval_results_full.csv"),"w",newline="") as f:  
    cw = csv.DictWriter(f,fields + ["topology"] ,delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    cw.writeheader()
    cw.writerows(eval_results_full_for_csv)
        
        
        
        
        
    
