import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import numpy as np
import csv
from utils import open_yaml, save_mask
from model_factories import trav_prior_factory
import os
from train_trav_seg import get_paths
from torchmetrics import JaccardIndex 
import matplotlib.pyplot as plt

"""
Measure performance of different models
"""
BENCHMARK_DIR = 'benchmark_results'


def test_seg_model(model, input_dim, base_src, base_trg, datasets, metric, exp_name):
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    src_paths, trg_paths = get_paths(base_src, base_trg, datasets)


    #GPU-WARM-UP
    x = torch.rand(1, 3, *input_dim).cuda()
    for _ in range(20):
        _ = model(x)

    
    ds = []
    times =[]
    precisions=[]
    recalls=[]
    with torch.no_grad():
        for i in range(len(src_paths)):
            img =  torchvision.io.read_image(src_paths[i]).float()/256

            trg = torch.load(trg_paths[i])
            
            _,x,y = img.size()

            img_interp = F.interpolate(img[None], input_dim, mode='bilinear').squeeze(0)

            starter.record()
            out = model(img_interp.cuda()[None]) 
            ender.record()

            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            times.append(curr_time)

            out_interp = F.interpolate(out, size=(x,y), mode='bilinear').detach().cpu()[0]
            #save_mask(img.permute(1,2,0).detach().cpu(), torch.argmax(out_interp[0], dim=0).detach().cpu(), 'hi.png')
            out_interp = F.softmax(out_interp, dim=0)

            p, r = get_pr(trg, out_interp)
            precisions.append(p)
            recalls.append(r)

            preds = torch.argmax(out_interp, dim=0)
            gt = torch.argmax(trg, dim=0)
            d = metric(preds, gt)
            ds.append(d)

            print(f'time: {curr_time:.2f} ms, mIoU: {d:.2f}')

    avg_precisions = torch.stack(precisions).nanmean(dim=0)
    avg_recalls = torch.stack(recalls).nanmean(dim=0)
    plt.plot(avg_recalls, avg_precisions)
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig(f'{BENCHMARK_DIR}/{exp_name}_pr.png')


    times = np.array(times)
    mean_syn = np.mean(times)
    std_syn = np.std(times)
    avg_metric = np.mean(np.array(ds))
    
    print(f'avg metric: {avg_metric}')
    print(f'avg time: {mean_syn}')
    print(f'stdev time: {std_syn}')
    
    with open(f'{BENCHMARK_DIR}/{exp_name}.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['mean jaccard (miou)', 'mean latency (ms)', 'st dev latency'])
        writer.writerow([avg_metric, mean_syn, std_syn])

def get_pr(y, y_hat):
    thresholds = torch.range(0, 1, .02)
    precisions = []
    recalls = []
    for threshold in thresholds:
        pred_positives = y_hat[1] >= threshold
        preds = torch.zeros_like(pred_positives)
        preds[pred_positives] = 1

        gt_positives = y[1] >= .5
        gt = torch.zeros_like(gt_positives)
        gt[gt_positives] = 1

        fp = torch.sum(preds * ~gt) # pred & ~gt
        tp = torch.sum(preds * gt) # pred & gt
        fn = torch.sum(~preds * gt) # ~pred & gt

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        precisions.append(precision)
        recalls.append(recall)

    p = torch.as_tensor(precisions)
    p = torch.nan_to_num(p, nan=1.0)
    r = torch.as_tensor(recalls)
    r = torch.nan_to_num(r, nan=0.0)

    return p, r
    




def get_fastervit_trav(cfg_path):

    cfg=  open_yaml(cfg_path)

    size = cfg['traversability']['img_dim']
    vpt = cfg['traversability']['vpt']
    vpt_prompt_length = cfg['traversability']['vpt_prompt_length']
    checkpoint = cfg['traversability']['checkpoint_path']

    model = trav_prior_factory(checkpoint, size, vpt, vpt_prompt_length, 'cuda')
    return model


def time_per_image(model, exp_name):

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 30
    timings=np.zeros((repetitions,1))

        
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            print(f'rep {rep}')
            x = torch.rand(1, 3, 224, 224)

            starter.record()
            _ = model(x)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)

    with open(f'{BENCHMARK_DIR}/{exp_name}.csv', 'w') as f:
        writer = csv.writer(f)
        writer.write_row(['mean latency (ms)', 'st dev time'])
        writer.write_row([mean_syn, std_syn])

    print(f'{1000 / mean_syn} hz' )

if __name__ == "__main__":
    data_base =  '/home/pcgta/Documents/playground/distill/full_data'
    data_ovseg_preds = '/home/pcgta/Documents/playground/distill/full_data_preds'
    test_datasets = ['anymal_site', 'anymal_lab']
    metric = JaccardIndex(task='multiclass', num_classes=2)

    fastervit_cfg = '/home/pcgta/Documents/playground/bc_trav/bc_trav/configs/tuned_fastervit.yaml'
    fastervit = get_fastervit_trav(fastervit_cfg)
    input_dim =  open_yaml(fastervit_cfg)['traversability']['img_dim']
    test_seg_model(fastervit, 
                   input_dim,
                    data_base,
                    data_ovseg_preds,
                    test_datasets, 
                    metric, 
                    exp_name='fastervit_miou')