import torch
import logging
import os.path as osp
from utils import AverageMeter
import numpy as np
from torch.nn.utils import clip_grad_norm_

from losses.relation_rectify import flexible_relation_kl_loss

logger = logging.getLogger(__name__)


class Engine:
    def __init__(
        self,
        seeds,
        args,
        model,
        model_ema,
        optim,
        lrs,
        loss,
        evaluator,
        dl_trn,
        dl_tst,
        tb_writer,
        savedir,
        old_model=None,
    ) -> None:
        self.seeds = seeds
        self.args = args
        self.model = model
        self.model_ema = model_ema
        self.optim = optim
        self.lrs = lrs
        self.loss = loss
        self.evaluator = evaluator
        self.dl_trn = dl_trn
        self.dl_tst = dl_tst
        self.tb_writer = tb_writer
        self.savedir = savedir

        # incremental learning
        self.label_offset = getattr(args, 'label_offset', 0)

        # anti-forgetting: relation rectification distillation
        self.old_model = old_model
        self.rectify_weight = getattr(args, 'rectify_weight', 0.0)
        self.rectify_tau = getattr(args, 'rectify_tau', 0.1)
        self.flexible_beta = getattr(args, 'flexible_beta', 0.1)
        self.use_adaptive_alpha = getattr(args, 'use_adaptive_alpha', 1)

        if self.old_model is not None:
            self.old_model.eval()
            for p in self.old_model.parameters():
                p.requires_grad_(False)

        if self.args.amp:
            logger.info("\nUsing Automatic Mixed Precision (AMP)")
            self.scaler = torch.cuda.amp.GradScaler(2 ** 10.0)

        self.nums_epoch = self.dl_trn.sampler.nums_epoch

        if len(self.args.eval_freq) == 1:
            self.eval_epochs = [self.args.eval_freq[0] for _ in range(self.args.epochs // self.args.eval_freq[0])]
            self.eval_epochs = np.cumsum(self.eval_epochs).tolist()
            if self.args.epochs % self.args.eval_freq[0]:
                self.eval_epochs.append(self.args.epochs)
        else:
            assert self.args.eval_freq[-1] == self.args.epochs
            self.eval_epochs = self.args.eval_freq

        self.epoch = 0
        self.iter_count = 0

    def train_one_epoch(self, show_nums=50):
        if self.args.freeze_bb and self.epoch == 0 and hasattr(self.model, 'freeze_backbone'):
            self.model.freeze_backbone()
            logger.info("Backbone is frozen")

        if self.args.freeze_bb and self.epoch == self.args.freeze_bb and hasattr(self.model, 'unfreeze_backbone'):
            self.model.unfreeze_backbone()
            logger.info("Backbone is unfrozen")

        if self.args.eval_bb and self.epoch == 0 and hasattr(self.model, 'eval_backbone'):
            assert self.args.eval_bb <= self.eval_epochs[0], "eval_bb should be <= first eval epoch"
            self.model.eval_backbone()
            logger.info("Backbone is in eval mode")

        if self.args.eval_bb and self.epoch == self.args.eval_bb and hasattr(self.model, 'train_backbone'):
            self.model.train_backbone()
            logger.info("Backbone is in train mode")

        losses_avg = AverageMeter()
        gradnorm_avg = AverageMeter()

        self.dl_trn.sampler.set(self.seeds[self.epoch])

        for i, (imgs, pids, cids) in enumerate(self.dl_trn):

            imgs = imgs.to('cuda')
            pids = pids.to('cuda') + self.label_offset

            if self.model.use_cid:
                cids_kwargs = {'cids': cids.to('cuda')}
            else:
                cids_kwargs = {}

            with torch.autocast(device_type='cuda', enabled=self.args.amp):

                # ===== current model forward =====
                outputs = self.model(imgs, **cids_kwargs)

                # ===== original losses =====
                loss_val, losses = self.loss(outputs, pids)

                # ==========================================================
                # Relation Rectification Distillation
                # ==========================================================
                if self.old_model is not None and self.rectify_weight > 0:

                    # ------------------------------------------------------
                    # current model features
                    # ------------------------------------------------------
                    feat_new_candidate = outputs[0]

                    if isinstance(feat_new_candidate, torch.Tensor):
                        feat_new = feat_new_candidate
                    else:
                        feat_new = torch.cat(feat_new_candidate, dim=1)

                    # ------------------------------------------------------
                    # old model features
                    # ------------------------------------------------------
                    with torch.no_grad():

                        # old model does NOT use cid
                        old_outputs = self.old_model(imgs)

                        if isinstance(old_outputs, torch.Tensor):
                            feat_old = old_outputs

                        elif isinstance(old_outputs, (list, tuple)):

                            feat_old_candidate = old_outputs[0]

                            if isinstance(feat_old_candidate, torch.Tensor):
                                feat_old = feat_old_candidate
                            else:
                                feat_old = torch.cat(feat_old_candidate, dim=1)

                        else:
                            raise TypeError(
                                f"Unsupported old_model output type: {type(old_outputs)}"
                            )

                    # ------------------------------------------------------
                    # relation rectify KL loss
                    # ------------------------------------------------------
                    loss_rect, aff_new, aff_old, aff_soft, same, diff = flexible_relation_kl_loss(
                        feat_new=feat_new.float(),
                        feat_old=feat_old.float(),
                        pids=pids,
                        tau=self.rectify_tau,
                        beta=self.flexible_beta,
                        return_affinity=True
                    )
                    # ------------------------------------------------------
                    # adaptive confidence weight
                    # ------------------------------------------------------
                    # with torch.no_grad():
                    #     # remove self relation
                    #     if same.dim() == 2 and same.size(0) == same.size(1):
                    #         same = same.clone()
                    #         same.fill_diagonal_(False)
                    #
                    #     pos_old = aff_old[same]
                    #     neg_old = aff_old[diff]
                    #     pos_new = aff_new[same]
                    #     neg_new = aff_new[diff]
                    #
                    #     if (
                    #             pos_old.numel() > 0 and
                    #             neg_old.numel() > 0 and
                    #             pos_new.numel() > 0 and
                    #             neg_new.numel() > 0
                    #     ):
                    #         quality_old = pos_old.mean() - neg_old.mean()
                    #         quality_new = pos_new.mean() - neg_new.mean()
                    #
                    #         conf_old = pos_old.mean()
                    #         conf_new = pos_new.mean()
                    #
                    #         alpha = 2.0 * conf_old / (conf_old + conf_new + 1e-6)
                    #         alpha = torch.clamp(alpha, 0.5, 1.5)
                    #
                    #     else:
                    #         quality_old = torch.tensor(0.0, device=imgs.device)
                    #         quality_new = torch.tensor(0.0, device=imgs.device)
                    #         alpha = torch.tensor(0.5, device=imgs.device)
                    with torch.no_grad():
                        # remove self relation
                        if same.dim() == 2 and same.size(0) == same.size(1):
                            same = same.clone()
                            same.fill_diagonal_(False)

                        pos_old = aff_old[same]
                        neg_old = aff_old[diff]
                        pos_new = aff_new[same]
                        neg_new = aff_new[diff]

                        if (
                                pos_old.numel() > 0 and
                                neg_old.numel() > 0 and
                                pos_new.numel() > 0 and
                                neg_new.numel() > 0
                        ):
                            quality_old = pos_old.mean() - neg_old.mean()
                            quality_new = pos_new.mean() - neg_new.mean()

                            conf_old = pos_old.mean()
                            conf_new = pos_new.mean()

                            alpha = 2.0 * conf_old / (conf_old + conf_new + 1e-6)
                            alpha = torch.clamp(alpha, 0.5, 1.5)

                        else:
                            quality_old = torch.tensor(0.0, device=imgs.device)
                            quality_new = torch.tensor(0.0, device=imgs.device)
                            alpha = torch.tensor(0.5, device=imgs.device)

                        # only use flexible topology, disable adaptive alpha
                        if not self.use_adaptive_alpha:
                            alpha = torch.tensor(1.0, device=imgs.device)

                        # only use flexible topology, disable adaptive alpha
                        if not self.use_adaptive_alpha:
                            alpha = torch.tensor(1.0, device=imgs.device)

                    # ------------------------------------------------------
                    # apply adaptive distillation
                    # ------------------------------------------------------
                    loss_val = (
                            loss_val
                            + self.rectify_weight * alpha * loss_rect
                    )
                    # ------------------------------------------------------
                    # logging
                    # ------------------------------------------------------
                    losses['rectify_kl'] = float(loss_rect.item())
                    losses['rectify_alpha'] = float(alpha.item())
                    losses['q_old'] = float(quality_old.item())
                    losses['q_new'] = float(quality_new.item())

            # ==============================================================
            # update average losses
            # ==============================================================
            losses_avg(losses)

            if torch.isnan(loss_val):
                return False

            if self.args.amp:
                self.scaler.scale(loss_val).backward()
                self.scaler.unscale_(self.optim)

                if self.args.grad_clip > 0:
                    norm = clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                    gradnorm_avg({'gradnorm': norm.item()})
                else:
                    norm = 0.0

                self.scaler.step(self.optim)
                self.scaler.update()
            else:
                loss_val.backward()

                if self.args.grad_clip > 0:
                    norm = clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                    gradnorm_avg({'gradnorm': norm.item()})
                else:
                    norm = 0.0

                self.optim.step()

            self.optim.zero_grad()

            if self.model_ema is not None:
                self.model_ema.update(self.model)

            if (i + 1) % show_nums == 0:
                logger.info(f"\tIter [{i+1}/{self.nums_epoch}] Losses: {losses_avg} Gradnorm: {gradnorm_avg}")

            if (i + 1) == self.nums_epoch:
                if self.args.lr_scheduler == 'TimmScheduler':
                    lrs = self.lrs._get_lr(self.epoch)[:2]
                else:
                    lrs = self.lrs.get_last_lr()[:2]

                if self.tb_writer is not None:
                    self.tb_writer.add_scalars(
                        'lrs',
                        {'lr-' + str(i): lr for i, lr in enumerate(lrs)},
                        self.epoch + 1
                    )

                logger.info(f"\tIter [{i+1}/{self.nums_epoch}] Losses: {losses_avg} Gradnorm: {gradnorm_avg}")
                logger.info(
                    f"Epoch [{self.epoch+1}/{self.args.epochs}] "
                    f"Lrs {[f'{lrs[i]:.4e}' for i in range(len(lrs))]} "
                    f"Losses: {losses_avg}"
                )

                if self.tb_writer is not None:
                    self.tb_writer.add_scalars('losses', losses_avg.avgs, self.epoch + 1)

            self.iter_count += 1

            if self.iter_count <= self.args.lr_scheduler_kwargs['warmup_iters'] and self.args.lr_scheduler == 'LinearWarmupLrScheduler':
                if self.iter_count % show_nums == 0 or self.iter_count == self.args.lr_scheduler_kwargs['warmup_iters'] or self.iter_count == 1:
                    logger.info(
                        f"\t\tIter [{self.iter_count}/{self.args.lr_scheduler_kwargs['warmup_iters']}] "
                        f"Warmup Lr: {self.lrs.get_last_lr()[0]}"
                    )
                self.lrs.step()

            if (i + 1) == self.nums_epoch and self.args.sp_forever:
                break

        return True

    def train_test(self):
        best_mAP = 0.0
        start_epoch = self.epoch
        self.model.train()

        for self.epoch in range(start_epoch, self.args.epochs):
            if not self.train_one_epoch(self.args.show_nums):
                raise ValueError("NaN loss encountered")

            if self.args.lr_scheduler == 'TimmScheduler':
                self.lrs.step(self.epoch)
            else:
                if self.iter_count > self.args.lr_scheduler_kwargs['warmup_iters']:
                    self.lrs.step()

            if (self.epoch + 1) in self.eval_epochs:
                metrics, metrics_flip = self.test(self.epoch)
                self.save_checkpoint(osp.join(self.savedir, f"ckpt_{self.epoch+1}.pth"))

                if metrics is not None and metrics['mAP'] > best_mAP:
                    best_mAP = metrics['mAP']

                if metrics_flip is not None and metrics_flip['mAP'] > best_mAP:
                    best_mAP = metrics_flip['mAP']

        if self.model_ema is not None:
            torch.save(self.model_ema.ema.state_dict(), osp.join(self.savedir, "ema_model.pth"))

    def test(self, epoch=-1, ckpt=None):
        if ckpt is not None:
            if 'model' in ckpt:
                self.model.load_state_dict(ckpt['model'])
            else:
                self.model.load_state_dict(torch.load(ckpt))

        metrics = self.evaluator(epoch + 1)
        self.model.train()

        if isinstance(metrics, tuple):
            return metrics

        if isinstance(metrics, list):
            if len(metrics) == 2:
                return metrics[0], metrics[1]
            if len(metrics) == 1:
                return metrics[0], None

        return metrics, None

    def save_checkpoint(self, checkpoint_dir: str):
        ckpt_path = osp.join(checkpoint_dir, "ckpt.pth") if '.pth' not in checkpoint_dir else checkpoint_dir
        ckpt = {
            'epoch': self.epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optim.state_dict(),
            'lr_scheduler': self.lrs.state_dict(),
            'scaler': self.scaler.state_dict() if self.args.amp else None
        }
        torch.save(ckpt, ckpt_path)

    def load_checkpoint(self, checkpoint_dir: str):
        ckpt_path = osp.join(checkpoint_dir, "ckpt.pth") if '.pth' not in checkpoint_dir else checkpoint_dir
        ckpt = torch.load(ckpt_path)
        self.epoch = ckpt['epoch']
        self.model.load_state_dict(ckpt['model'])
        self.optim.load_state_dict(ckpt['optimizer'])
        self.lrs.load_state_dict(ckpt['lr_scheduler'])
        if self.args.amp and ckpt['scaler'] is not None:
            self.scaler.load_state_dict(ckpt['scaler'])