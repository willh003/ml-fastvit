import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import numpy as np
import csv
from utils import open_yaml, save_mask
from train_trav_seg import get_paths
from torchmetrics import JaccardIndex 
import matplotlib.pyplot as plt
import os
import tqdm

"""
Measure performance of different models
"""
BENCHMARK_DIR = 'benchmark_results'


def test_seg_model(model, input_dim, src_paths, trg_paths, metric):
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)



    #GPU-WARM-UP
    x = torch.rand(1, 3, *input_dim).cuda()
    for _ in range(20):
        _ = model(x)

    
    ds = []
    times =[]
    precisions=[]
    recalls=[]
    with torch.no_grad():
        with tqdm.tqdm(range(len(src_paths))) as t:
            for i in t:
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
                #out_interp = F.softmax(out_interp, dim=0)


                p, r = get_pr(trg, out_interp)
                precisions.append(p)
                recalls.append(r)

                preds = torch.argmax(out_interp, dim=0)
                gt = torch.argmax(trg, dim=0)
                d = metric(preds, gt)
                ds.append(d)

                t.set_description(f'time: {curr_time:.2f} ms, mIoU: {d:.2f}')

    avg_precisions = torch.stack(precisions).nanmean(dim=0)
    avg_recalls = torch.stack(recalls).nanmean(dim=0)

    times = torch.as_tensor(times)
    mean_syn = torch.mean(times).item()
    std_syn = torch.std(times).item()
    ds = torch.as_tensor(ds)
    avg_metric = torch.mean(ds).item()
    
    print(f'avg metric: {avg_metric}')
    print(f'avg time: {mean_syn}')
    print(f'stdev time: {std_syn}')
    return avg_precisions, avg_recalls, times, ds


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
    

def get_fastervit_trav(cfg_path , ckpt=None):
    from model_factories import trav_prior_factory

    cfg=  open_yaml(cfg_path)

    size = cfg['traversability']['img_dim']
    vpt = cfg['traversability']['vpt']
    vpt_prompt_length = cfg['traversability']['vpt_prompt_length']
    
    if ckpt is None:
        ckpt = cfg['traversability']['checkpoint_path']

    model = trav_prior_factory(ckpt, size, vpt, vpt_prompt_length, 'cuda')
    return model

def get_scnn_trav(cfg_path, ckpt):
    from model_factories import fast_scnn_factory

    cfg=  open_yaml(cfg_path)

    size = cfg['traversability']['img_dim']

    model = fast_scnn_factory(ckpt, size, 'cuda')
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

def benchmark_all_models(ckpts, input_dim, src_paths, trg_paths, metric, exp_name):

    all_precision = []
    all_recall = []
    all_times = []
    all_ds = []
    for ckpt in ckpts:
        if 'scnn' in ckpt:
            model = get_scnn_trav('/home/pcgta/Documents/bc_trav/bc_trav/configs/tuned_fastscnn.yaml', ckpt)
        elif 'fastervit' in ckpt:
            model = get_fastervit_trav('/home/pcgta/Documents/bc_trav/bc_trav/configs/tuned_fastervit.yaml', ckpt)
        else:
            raise Exception('Unclear what type of model this is - put fastervit or scnn in the checkpoint name')

        precision, recall, times, ds = test_seg_model(model, 
                                                    input_dim,
                                                    src_paths,
                                                    trg_paths,
                                                    metric)
        name = ckpt.split('/')[-1].split('.')[0]
        for i, arr in enumerate([precision, recall, times ,ds]):
            torch.save(arr, f'{BENCHMARK_DIR}/{i}_metric_{name}.pt')
        
        all_precision.append(precision.cpu())
        all_recall.append(recall.cpu())
        all_times.append(times.cpu())
        all_ds.append(ds.cpu())
    
def plot_data(ckpts, ckpt_labels, exp_name):

    with open(f'{BENCHMARK_DIR}/{exp_name}.csv', 'w') as f:
        writer = csv.writer(f)
        
        all_ds = []
        for ckpt, label in zip(ckpts, ckpt_labels):
            name = ckpt.split('/')[-1].split('.')[0]
            
            pr = torch.load(f'{BENCHMARK_DIR}/0_metric_{name}.pt')
            re = torch.load(f'{BENCHMARK_DIR}/1_metric_{name}.pt')

            #plt.plot(re, pr, label=label)

            if os.path.exists(f'{BENCHMARK_DIR}/2_metric_{name}.pt'):
                times = torch.load(f'{BENCHMARK_DIR}/2_metric_{name}.pt')
                ds = torch.load(f'{BENCHMARK_DIR}/3_metric_{name}.pt')

                mean_syn = torch.mean(times).item()
                std_syn = torch.std(times).item()
                avg_metric = torch.mean(ds).item()
                all_ds.append(ds)

                #writer.writerow(['model','mean jaccard (miou)', 'mean latency (ms)', 'st dev latency'])
                #writer.writerow([label, avg_metric, mean_syn, std_syn])

        all_ds = torch.stack(all_ds)
        breakpoint()
        minmax = torch.argmax(torch.min(all_ds, dim=0))


    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.savefig(f'{BENCHMARK_DIR}/{exp_name}_pr.png')


def run_on_one(path):
    ckpt = '/home/pcgta/Documents/bc_trav/bc_trav/trav_checkpoints/scnn-epoch=24-step=19225-b=8-lr=6e-04recon_kitti_asrl.ckpt'
    model = get_scnn_trav('/home/pcgta/Documents/bc_trav/bc_trav/configs/tuned_fastscnn.yaml', ckpt)
    img =  torchvision.io.read_image(path).float()/256

    _,x,y = img.size()

    img_interp = F.interpolate(img[None], (224, 224), mode='bilinear').squeeze(0)

    out = model(img_interp.cuda()[None]) 


    out_interp = F.interpolate(out, size=(x,y), mode='bilinear').detach().cpu()[0]

    plt.imshow(img.permute(1,2,0))
    plt.imshow(torch.argmax(out_interp, dim=0), cmap='jet', alpha=.3)

    save_path = path.split('.')[0] + '-mask-scnn-partial.png'
    plt.savefig(save_path)


if __name__ == "__main__":
    path = '/home/pcgta/Documents/bc_trav/bc_trav/DEMO_IMAGES/000003-worst.png'
    run_on_one(path)
    breakpoint()


    if not os.path.exists(BENCHMARK_DIR):
        os.mkdir(BENCHMARK_DIR)

    data_base =  '/home/pcgta/Documents/bc_trav/bc_trav/distill_data/full_data'
    data_teacher_preds = '/home/pcgta/Documents/bc_trav/bc_trav/distill_data/full_data_preds'
    test_datasets = ['anymal_site', 'anymal_lab']
    metric = JaccardIndex(task='multiclass', num_classes=2)
    src_paths, trg_paths = get_paths(data_base, data_teacher_preds, test_datasets)


    fastervit_cfg = '/home/pcgta/Documents/bc_trav/bc_trav/configs/tuned_fastervit.yaml'
    #fastervit = get_fastervit_trav(fastervit_cfg, ckpt = '/home/pcgta/Documents/bc_trav/bc_trav/trav_checkpoints/fastervit-epoch=9-b=16-lr=6e-03recon_sacson_kitti_asrl (copy).ckpt')
    input_dim =  open_yaml(fastervit_cfg)['traversability']['img_dim']
    
    scnn_cfg = '/home/pcgta/Documents/bc_trav/bc_trav/configs/tuned_fastscnn.yaml'


    ckpts = [
        # '/home/pcgta/Documents/bc_trav/bc_trav/trav_checkpoints/fastervit-epoch=28-b=16-lr=6e-03recon_sacson_kitti_asrl.ckpt',
        # '/home/pcgta/Documents/bc_trav/bc_trav/trav_checkpoints/fastervit-epoch=44-lr=6e-3recon_kitti_asrl.ckpt',
    #    '/home/pcgta/Documents/bc_trav/bc_trav/trav_checkpoints/scnn-epoch=14-b=8-lr=6e-03sacson_recon_kitti_asrl.ckpt',
        '/home/pcgta/Documents/bc_trav/bc_trav/trav_checkpoints/scnn-epoch=19-b=8-lr=6e-04sacson_recon_kitti_asrl.ckpt',
        '/home/pcgta/Documents/bc_trav/bc_trav/trav_checkpoints/scnn-epoch=24-step=19225-b=8-lr=6e-04recon_kitti_asrl.ckpt'
    ]


    benchmark_all_models(ckpts, 
                   input_dim,
                    src_paths,
                    trg_paths,
                    metric,
                    'full_benchmark_3')

    # plot_data(ckpts, ['ViT-S', 'ViT-L', 'SCNN-S', 'SCNN-L'], exp_name='all_data')