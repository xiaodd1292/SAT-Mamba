import os
import copy
import torch
import argparse
from functools import partial
from tabulate import tabulate

from utils import setup, str2list, str2dict
from data import build_dataloaders
from model import build_model
from losses import Loss
from optims import build_optimizer
from evaluation import Evaluator
from engine import Engine
from timm.utils import ModelEma


def build_parser():
    parser = argparse.ArgumentParser()

    # system
    parser.add_argument('--gpus', type=str2list, default='0')
    parser.add_argument('--exp', type=str, default='continual_reidmamba')
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--config', type=str, default='')

    # data
    parser.add_argument('--dataroot', type=str, default='path/to/your/dataset')
    parser.add_argument(
        '--dataset',
        type=str2list,
        default='VesselReID,CMshipReID',
        help='incremental order'
    )
    parser.add_argument(
        '--unseen_dataset',
        type=str2list,
        default='Warships-ReID,Sub-MARVEL',
        help='unseen datasets for evaluation only'
    )
    parser.add_argument('--p_trn', type=float, default=1.0)
    parser.add_argument('--split_mode_trn', type=str, default='by_person')
    parser.add_argument('--dataset_trn', type=str2list, default='')
    parser.add_argument('--dataset_qry', type=str2list, default='')
    parser.add_argument('--dataset_gal', type=str2list, default='')
    parser.add_argument(
        '--pixel_mean',
        type=partial(str2list, f=float),
        default='0.485,0.456,0.406'
    )
    parser.add_argument(
        '--pixel_std',
        type=partial(str2list, f=float),
        default='0.229,0.224,0.225'
    )
    parser.add_argument(
        '--img_size',
        type=partial(str2list, f=int),
        default="256,256"
    )
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument(
        '--bs_trn',
        type=partial(str2list, f=int),
        default='16,4'
    )
    parser.add_argument(
        '--bs_tst',
        type=partial(str2list, f=int),
        default='128'
    )
    parser.add_argument('--sp_forever', type=bool, default=True)
    parser.add_argument('--sp_seeds', type=bool, default=False)
    parser.add_argument(
        '--aa_tf',
        type=str2dict,
        default='do_aa(bool)=0|aa_prob(float)=0.1'
    )
    parser.add_argument(
        '--crop_tf',
        type=str2dict,
        default='do_crop(bool)=0|crop_size(int)=[384,128]|crop_scale(float)=[0.08,1.0]|crop_ratio(float)=[0.75,1.33]'
    )
    parser.add_argument(
        '--pad_tf',
        type=str2dict,
        default='do_pad(bool)=1|padding_size(int)=[10,10]|padding_mode(str)=constant|padding_fill(float)=[0,0,0]'
    )
    parser.add_argument(
        '--flip_tf',
        type=str2dict,
        default='do_flip(bool)=1|flip_prob(float)=0.5'
    )
    parser.add_argument(
        '--rea_tf',
        type=str2dict,
        default='do_rea(bool)=1|rea_prob(float)=0.5|rea_value(str)=random|rea_scale(float)=[0.02,0.4]|rea_ratio(float)=[0.3,3.33]'
    )

    # testing
    parser.add_argument('--eval_freq', type=partial(str2list, f=int), default='20')
    parser.add_argument('--show_nums', type=int, default=50)
    parser.add_argument('--dist_metric', type=str, default='cosine')
    parser.add_argument('--use_cython', type=bool, default=True)
    parser.add_argument('--test_flip', type=bool, default=True)
    parser.add_argument('--search_options', type=partial(str2list, f=int), default='3,2')
    parser.add_argument('--rerank', type=bool, default=False)
    parser.add_argument('--rerank_k1', type=int, default=20)
    parser.add_argument('--rerank_k2', type=int, default=6)
    parser.add_argument('--rerank_lambda', type=float, default=0.3)

    # model
    parser.add_argument('--model', type=str, default='ReIDMamba')
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument(
        '--model_kwargs',
        type=str2dict,
        default='backbone_name(str)=mambar_small_patch16_224|drop_path_rate(float)=0.3|num_cls_tokens(int)=12|cls_reduce(int)=4|use_cid(bool)=1|stride_size(int)=16|num_branches(int)=3|token_fusion_type(str)=max'
    )
    parser.add_argument(
        '--ema',
        type=str2dict,
        default='ema_model(bool)=0|ema_decay(float)=0.9992'
    )

    # loss
    parser.add_argument(
        '--loss',
        type=str2list,
        default='triplet_loss,cross_entropy_loss,ratr_intra_loss,ratr_inter_loss'
    )
    parser.add_argument(
        '--loss_weights',
        type=partial(str2list, f=float),
        default='1.0,1.0,1.0,1.0'
    )
    parser.add_argument(
        '--loss_nums',
        type=partial(str2list, f=int),
        default='3,3,1,1'
    )
    parser.add_argument(
        '--loss_kwargs',
        type=partial(str2list, f=str2dict),
        default='margin(float)=1.2,label_smoothing(float)=0.1,N(int)=3|PK(int)=[16,4]|tau(float)=0.1,N(int)=3|PK(int)=[16,4]|tau(float)=0.1'
    )

    # optimizer
    parser.add_argument('--optim', type=str, default='sgd')
    parser.add_argument(
        '--optim_kwargs',
        type=str2dict,
        default='lr(float)=0.008|weight_decay(float)=0.0|momentum(float)=0.9|nesterov(bool)=0'
    )
    parser.add_argument('--lr_scheduler', type=str, default='LinearWarmupLrScheduler')
    parser.add_argument(
        '--lr_scheduler_kwargs',
        type=str2dict,
        default='warmup_epochs(int)=4|warmup_iters(int)=1200|lr_multiplier(float)=1e-2|lrs2(str)=CosineAnnealingLR|lrs2_kwargs(str2dict)={T_max(int)=155|eta_min(float)=8e-6}'
    )
    parser.add_argument('--epochs', type=int, default=160)
    parser.add_argument('--freeze_bb', type=int, default=0)
    parser.add_argument('--eval_bb', type=int, default=0)
    parser.add_argument('--amp', type=bool, default=True)
    parser.add_argument('--grad_clip', type=float, default=10.0)

    # continual learning
    parser.add_argument('--label_offset', type=int, default=0)
    parser.add_argument('--cid_offset', type=int, default=0)
    # anti-forgetting: relation rectification distillation
    parser.add_argument('--rectify_weight', type=float, default=10.0)
    parser.add_argument('--rectify_tau', type=float, default=0.01)
    parser.add_argument('--flexible_beta', type=float, default=0.002)
    # whether to use adaptive distillation weight alpha
    parser.add_argument('--use_adaptive_alpha', type=int, default=1)
    # historical-current adaptive feature coupling
    parser.add_argument('--use_hc_coupling', type=int, default=1)
    parser.add_argument('--coupling_hidden_ratio', type=float, default=0.5)
    # discrepancy-aware patch token interaction
    parser.add_argument('--use_token_swap', type=int, default=1)
    parser.add_argument('--swap_ratio', type=float, default=0.04)


    return parser


def evaluate_dataset_group(
    args,
    model,
    dataset_names,
    cid_offsets,
    tb_writer,
    logger,
    group_name='Seen',
    disable_sie=False,
):
    rows_ori = []
    rows_flip = []

    ori_results = {}
    flip_results = {}

    if dataset_names is None or len(dataset_names) == 0:
        return ori_results, flip_results, None, None

    old_use_cid = getattr(model, 'use_cid', False)

    for name in dataset_names:
        args_eval = copy.deepcopy(args)
        args_eval.dataset = [name]
        args_eval.label_offset = 0

        if disable_sie:
            args_eval.cid_offset = 0
            model.use_cid = False
        else:
            args_eval.cid_offset = cid_offsets.get(name, 0)

        _, dl_tst = build_dataloaders(args_eval)

        evaluator = Evaluator(args_eval, model, dl_tst, tb_writer)
        eval_out = evaluator(-1)

        if isinstance(eval_out, (list, tuple)):
            ori_metrics = eval_out[0] if len(eval_out) > 0 else None
            flip_metrics = eval_out[1] if len(eval_out) > 1 else None
        elif isinstance(eval_out, dict):
            ori_metrics = eval_out
            flip_metrics = None
        else:
            raise TypeError(f"Unsupported evaluator output type: {type(eval_out)}")

        if ori_metrics is not None:
            ori_results[name] = ori_metrics
            rows_ori.append([
                name,
                f"{ori_metrics['Rank-1']:.2f}",
                f"{ori_metrics['mAP']:.2f}",
                f"{ori_metrics['mINP']:.2f}",
            ])

        if flip_metrics is not None:
            flip_results[name] = flip_metrics
            rows_flip.append([
                name,
                f"{flip_metrics['Rank-1']:.2f}",
                f"{flip_metrics['mAP']:.2f}",
                f"{flip_metrics['mINP']:.2f}",
            ])

    model.use_cid = old_use_cid

    def compute_avg(results_dict):
        if len(results_dict) == 0:
            return None

        avg_r1 = sum(v['Rank-1'] for v in results_dict.values()) / len(results_dict)
        avg_map = sum(v['mAP'] for v in results_dict.values()) / len(results_dict)
        avg_minp = sum(v['mINP'] for v in results_dict.values()) / len(results_dict)

        return avg_r1, avg_map, avg_minp

    avg_ori = compute_avg(ori_results)
    avg_flip = compute_avg(flip_results)

    if avg_ori is not None:
        rows_ori.append([
            f"{group_name}-Avg",
            f"{avg_ori[0]:.2f}",
            f"{avg_ori[1]:.2f}",
            f"{avg_ori[2]:.2f}",
        ])

        logger.info(f"\n[{group_name} Evaluation - ORIGINAL]")
        logger.info("\n" + tabulate(
            rows_ori,
            headers=['Dataset', 'Rank-1', 'mAP', 'mINP'],
            tablefmt='orgtbl'
        ))

    if avg_flip is not None:
        rows_flip.append([
            f"{group_name}-Avg",
            f"{avg_flip[0]:.2f}",
            f"{avg_flip[1]:.2f}",
            f"{avg_flip[2]:.2f}",
        ])

        logger.info(f"\n[{group_name} Evaluation - FLIPPED]")
        logger.info("\n" + tabulate(
            rows_flip,
            headers=['Dataset', 'Rank-1', 'mAP', 'mINP'],
            tablefmt='orgtbl'
        ))

    return ori_results, flip_results, avg_ori, avg_flip


def continual_train(args):
    tb_writer, logger, savedir, seeds = setup(args, determenistic=True, benchmark=True)

    train_order = copy.deepcopy(args.dataset)
    assert isinstance(train_order, list) and len(train_order) >= 1, "dataset must be a list for incremental order"

    logger.info(f"Incremental training order: {train_order}")

    model = None
    old_model = None

    global_num_classes = 0
    global_num_cids = 0

    cid_offsets = {}
    stage_summary = []

    for stage, dataset_name in enumerate(train_order):
        logger.info(f"\n========== Stage {stage+1}/{len(train_order)} : {dataset_name} ==========")

        args_stage = copy.deepcopy(args)
        args_stage.dataset = [dataset_name]
        args_stage.label_offset = global_num_classes
        args_stage.cid_offset = global_num_cids

        dl_trn, dl_tst = build_dataloaders(args_stage)

        num_new_classes = dl_trn.num_cls
        num_new_cids = dl_trn.num_cid

        cid_offsets[dataset_name] = global_num_cids

        if stage == 0:
            model = build_model(args_stage, num_new_classes, num_new_cids)
            model.to('cuda')
            global_num_classes = num_new_classes
            global_num_cids = num_new_cids
        else:
            model.expand_classifier(num_new_classes)

            if getattr(model, 'use_cid', False):
                model.expand_sie_embed(num_new_cids)

            global_num_classes += num_new_classes
            global_num_cids += num_new_cids

        if args_stage.ema['ema_model']:
            model_ema = ModelEma(model, decay=args_stage.ema['ema_decay'])
            logger.info(f'Using EMA with decay {args_stage.ema["ema_decay"]}')
        else:
            model_ema = None

        optim, lrs = build_optimizer(args_stage, model)

        loss = Loss(args_stage)
        loss.to('cuda')

        evaluator = Evaluator(args_stage, model, dl_tst, tb_writer)

        stage_savedir = os.path.join(savedir, f"stage_{stage+1}_{dataset_name}")
        os.makedirs(stage_savedir, exist_ok=True)

        eng = Engine(
            seeds=seeds,
            args=args_stage,
            model=model,
            model_ema=model_ema,
            optim=optim,
            lrs=lrs,
            loss=loss,
            evaluator=evaluator,
            dl_trn=dl_trn,
            dl_tst=dl_tst,
            tb_writer=tb_writer,
            savedir=stage_savedir,
            old_model=old_model,
        )

        logger.info("Starting stage training...")
        eng.train_test()

        # seen-dataset evaluation
        seen_datasets = train_order[:stage + 1]
        logger.info(f"\nSeen-dataset evaluation after stage {stage + 1}:")

        ori_results, flip_results, seen_avg_ori, seen_avg_flip = evaluate_dataset_group(
            args=args,
            model=model,
            dataset_names=seen_datasets,
            cid_offsets=cid_offsets,
            tb_writer=tb_writer,
            logger=logger,
            group_name='Seen',
            disable_sie=False,
        )

        # unseen-dataset evaluation
        unseen_datasets = []
        if hasattr(args, 'unseen_dataset') and args.unseen_dataset:
            unseen_datasets = copy.deepcopy(args.unseen_dataset)

        unseen_avg_ori = None
        unseen_avg_flip = None

        if len(unseen_datasets) > 0:
            logger.info(f"\nUnseen-dataset evaluation after stage {stage + 1}:")

            _, _, unseen_avg_ori, unseen_avg_flip = evaluate_dataset_group(
                args=args,
                model=model,
                dataset_names=unseen_datasets,
                cid_offsets=cid_offsets,
                tb_writer=tb_writer,
                logger=logger,
                group_name='Unseen',
                disable_sie=True,
            )

        # Final summary uses ORIGINAL by default
        seen_r1 = seen_avg_ori[0] if seen_avg_ori is not None else 0.0
        seen_map = seen_avg_ori[1] if seen_avg_ori is not None else 0.0
        seen_minp = seen_avg_ori[2] if seen_avg_ori is not None else 0.0

        if unseen_avg_ori is not None:
            unseen_r1 = f"{unseen_avg_ori[0]:.2f}"
            unseen_map = f"{unseen_avg_ori[1]:.2f}"
            unseen_minp = f"{unseen_avg_ori[2]:.2f}"
        else:
            unseen_r1 = "-"
            unseen_map = "-"
            unseen_minp = "-"

        stage_summary.append([
            stage + 1,
            dataset_name,
            f"{seen_r1:.2f}",
            f"{seen_map:.2f}",
            f"{seen_minp:.2f}",
            unseen_r1,
            unseen_map,
            unseen_minp,
        ])

        # save stage model
        model_path = os.path.join(stage_savedir, f"{dataset_name}_stage{stage+1}.pth")
        torch.save(model.state_dict(), model_path)
        logger.info(f"Saved stage model to: {model_path}")

        # update old model for next stage
        old_model = copy.deepcopy(model).to('cuda').eval()
        for p in old_model.parameters():
            p.requires_grad_(False)

    logger.info("\n========== Final Incremental Summary ==========")
    logger.info("\n" + tabulate(
        stage_summary,
        headers=[
            'Stage',
            'TrainOn',
            'Seen-Avg Rank-1',
            'Seen-Avg mAP',
            'Seen-Avg mINP',
            'Unseen-Avg Rank-1',
            'Unseen-Avg mAP',
            'Unseen-Avg mINP',
        ],
        tablefmt='orgtbl'
    ))


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    continual_train(args)