"""
A generic training script that works with any model and dataset.

Author: Paul-Edouard Sarlin (skydes)
"""

# Filter annoying warnings
import warnings

warnings.simplefilter("ignore", UserWarning)

import argparse
import copy
import re
import shutil
import signal
from collections import defaultdict
from pathlib import Path
from pydoc import locate

import numpy as np
import torch
from hydra import compose, initialize
from omegaconf import OmegaConf
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from siclib import __module_name__, logger
from siclib.datasets import get_dataset
from siclib.eval import run_benchmark
from siclib.models import get_model
from siclib.settings import EVAL_PATH, TRAINING_PATH
from siclib.utils.experiments import get_best_checkpoint, get_last_checkpoint, save_experiment
from siclib.utils.stdout_capturing import capture_outputs
from siclib.utils.summary_writer import SummaryWriter
from siclib.utils.tensor import batch_to_device
from siclib.utils.tools import (
    AverageMetric,
    MedianMetric,
    PRMetric,
    RecallMetric,
    fork_rng,
    get_device,
    set_seed,
)

# flake8: noqa
# mypy: ignore-errors


# TODO: Fix pbar pollution in logs
# TODO: add plotting during evaluation

default_train_conf = {
    "device": "0",  # 指定训练设备号
    "seed": "???",  # training seed
    "epochs": 1,  # number of epochs
    "num_steps": None,  # number of steps, overwrites epochs
    "optimizer": "adam",  # name of optimizer in [adam, sgd, rmsprop]
    "opt_regexp": None,  # regular expression to filter parameters to optimize
    "optimizer_options": {},  # optional arguments passed to the optimizer
    "lr": 0.001,  # learning rate
    "lr_schedule": {
        "type": None,
        "start": 0,
        "exp_div_10": 0,
        "on_epoch": False,
        "factor": 1.0,
    },
    "lr_scaling": [(100, ["dampingnet.const"])],
    "eval_every_iter": 1000,  # interval for evaluation on the validation set
    "save_every_iter": 5000,  # interval for saving the current checkpoint
    "log_every_iter": 200,  # interval for logging the loss to the console
    "log_grad_every_iter": None,  # interval for logging gradient hists
    "writer": "tensorboard",  # tensorboard or wandb
    "test_every_epoch": 1,  # interval for evaluation on the test benchmarks
    "keep_last_checkpoints": 10,  # keep only the last X checkpoints
    "load_experiment": None,  # initialize the model from a previous experiment
    "median_metrics": [],  # add the median of some metrics
    "recall_metrics": {},  # add the recall of some metrics
    "pr_metrics": {},  # add pr curves, set labels/predictions/mask keys
    "best_key": "loss/total",  # key to use to select the best checkpoint
    "dataset_callback_fn": None,  # data func called at the start of each epoch
    "dataset_callback_on_val": False,  # call data func on val data?
    "clip_grad": None,
    "pr_curves": {},
    "plot": None,
    "submodules": [],
}
default_train_conf = OmegaConf.create(default_train_conf)


def get_lr_scheduler(optimizer, conf):
    """Get lr scheduler specified by conf."""
    # logger.info(f"Using lr scheduler with conf: {conf}")
    if conf.type not in ["factor", "exp", None]:
        if hasattr(conf.options, "schedulers"):
            # Add option to chain multiple schedulers together
            # This is useful for e.g. warmup, then cosine decay
            """Example: {
                "type": "SequentialLR",
                "options": {
                    "milestones": [1_000],
                    "schedulers": [
                        {"type": "LinearLR", "options": {"total_iters": 10, "start_factor": 0.001}},
                        {"type": "MultiStepLR", "options": {"milestones": [40, 60], "gamma": 0.1}},
                    ],
                }
            }
            """
            schedulers = []
            for scheduler_conf in conf.options.schedulers:
                scheduler = get_lr_scheduler(optimizer, scheduler_conf)
                schedulers.append(scheduler)

            options = {k: v for k, v in conf.options.items() if k != "schedulers"}
            return getattr(torch.optim.lr_scheduler, conf.type)(optimizer, schedulers, **options)

        return getattr(torch.optim.lr_scheduler, conf.type)(optimizer, **conf.options)

    # backward compatibility
    def lr_fn(it):  # noqa: E306
        if conf.type is None:
            return 1
        if conf.type == "factor":
            return 1.0 if it < conf.start else conf.factor
        if conf.type == "exp":
            gam = 10 ** (-1 / conf.exp_div_10)
            return 1.0 if it < conf.start else gam
        else:
            raise ValueError(conf.type)

    return torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_fn)


@torch.no_grad()
def do_evaluation(model, loader, device, loss_fn, conf, pbar=True):
    model.eval()
    results = {}
    recall_results = {}
    pr_metrics = defaultdict(PRMetric)
    figures = []
    if conf.plot is not None:
        n, plot_fn = conf.plot
        plot_ids = np.random.choice(len(loader), min(len(loader), n), replace=False)
    for i, data in enumerate(tqdm(loader, desc="Evaluation", ascii=True, disable=not pbar)):
        data = batch_to_device(data, device, non_blocking=True)
        with torch.no_grad():
            pred = model(data)
            losses, metrics = loss_fn(pred, data)
            if conf.plot is not None and i in plot_ids:
                figures.append(locate(plot_fn)(pred, data))
            # add PR curves
            for k, v in conf.pr_curves.items():
                pr_metrics[k].update(
                    pred[v["labels"]],
                    pred[v["predictions"]],
                    mask=pred[v["mask"]] if "mask" in v.keys() else None,
                )
            del pred, data

        numbers = {**metrics, **{f"loss/{k}": v for k, v in losses.items()}}
        for k, v in numbers.items():
            if k not in results:
                results[k] = AverageMetric()
                if k in conf.median_metrics:
                    results[f"{k}_median"] = MedianMetric()

            if k not in recall_results and k in conf.recall_metrics.keys():
                ths = conf.recall_metrics[k]
                recall_results[k] = RecallMetric(ths)

            results[k].update(v)
            if k in conf.median_metrics:
                results[f"{k}_median"].update(v)
            if k in conf.recall_metrics.keys():
                recall_results[k].update(v)

        del numbers

    results = {k: results[k].compute() for k in results}

    for k, v in recall_results.items():
        for th, recall in zip(conf.recall_metrics[k], v.compute()):
            results[f"{k}_recall@{th}"] = recall

    return results, {k: v.compute() for k, v in pr_metrics.items()}, figures


def filter_parameters(params, regexp):
    """Filter trainable parameters based on regular expressions."""

    # Examples of regexp:
    #     '.*(weight|bias)$'
    #     'cnn\.(enc0|enc1).*bias'
    def filter_fn(x):
        n, p = x
        match = re.search(regexp, n)
        if not match:
            p.requires_grad = False
        return match

    params = list(filter(filter_fn, params))
    assert len(params) > 0, regexp
    logger.info("Selected parameters:\n" + "\n".join(n for n, p in params))
    return params


def pack_lr_parameters(params, base_lr, lr_scaling):
    """Pack each group of parameters with the respective scaled learning rate."""
    filters, scales = tuple(zip(*[(n, s) for s, names in lr_scaling for n in names]))
    scale2params = defaultdict(list)
    for n, p in params:
        scale = 1
        is_match = [f in n for f in filters]
        if any(is_match):
            scale = scales[is_match.index(True)]
        scale2params[scale].append((n, p))
    logger.info(
        "Parameters with scaled learning rate:\n%s",
        {s: [n for n, _ in ps] for s, ps in scale2params.items() if s != 1},
    )
    return [
        {"lr": scale * base_lr, "params": [p for _, p in ps]} for scale, ps in scale2params.items()
    ]


def training(rank, conf, output_dir, args):
    if args.restore:
        logger.info(f"Restoring from previous training of {args.experiment}")
        try:
            init_cp = get_last_checkpoint(args.experiment, allow_interrupted=False)
        except AssertionError:
            init_cp = get_best_checkpoint(args.experiment)
        logger.info(f"Restoring from checkpoint {init_cp.name}")
        init_cp = torch.load(str(init_cp), map_location="cpu")
        conf = OmegaConf.merge(OmegaConf.create(init_cp["conf"]), conf)
        conf.train = OmegaConf.merge(default_train_conf, conf.train)
        epoch = init_cp["epoch"] + 1

        # get the best loss or eval metric from the previous best checkpoint
        best_cp = get_best_checkpoint(args.experiment)
        best_cp = torch.load(str(best_cp), map_location="cpu")
        best_eval = best_cp["eval"][conf.train.best_key]
        del best_cp
    else:
        # we start a new, fresh training
        conf.train = OmegaConf.merge(default_train_conf, conf.train)
        epoch = 0
        best_eval = float("inf")
        if conf.train.load_experiment:
            logger.info(f"Will fine-tune from weights of {conf.train.load_experiment}")
            # the user has to make sure that the weights are compatible
            try:
                init_cp = get_last_checkpoint(conf.train.load_experiment)
            except AssertionError:
                init_cp = get_best_checkpoint(conf.train.load_experiment)
            # init_cp = get_last_checkpoint(conf.train.load_experiment)
            init_cp = torch.load(str(init_cp), map_location="cpu")
            # load the model config of the old setup, and overwrite with current config
            conf.model = OmegaConf.merge(OmegaConf.create(init_cp["conf"]).model, conf.model)
            print(conf.model)
        else:
            init_cp = None

    OmegaConf.set_struct(conf, True)  # prevent access to unknown entries
    set_seed(conf.train.seed)
    if rank == 0:
        writer = SummaryWriter(conf, args, str(output_dir))

    data_conf = copy.deepcopy(conf.data)
    if args.distributed:
        logger.info(f"Training in distributed mode with {args.n_gpus} GPUs")
        assert torch.cuda.is_available()
        device = rank
        torch.distributed.init_process_group(
            backend="nccl",
            world_size=args.n_gpus,
            rank=device,
            init_method="file://" + str(args.lock_file),
        )
        torch.cuda.set_device(device)

        # adjust batch size and num of workers since these are per GPU
        if "batch_size" in data_conf:
            data_conf.batch_size = int(data_conf.batch_size / args.n_gpus)
        if "train_batch_size" in data_conf:
            data_conf.train_batch_size = int(data_conf.train_batch_size / args.n_gpus)
        if "num_workers" in data_conf:
            data_conf.num_workers = int((data_conf.num_workers + args.n_gpus - 1) / args.n_gpus)
    else:
        device = get_device(conf.device)
    logger.info(f"Using device {device}")

    dataset = get_dataset(data_conf.name)(data_conf)

    # Optionally load a different validation dataset than the training one
    val_data_conf = conf.get("data_val", None)
    if val_data_conf is None:
        val_dataset = dataset
    else:
        val_dataset = get_dataset(val_data_conf.name)(val_data_conf)

    # @TODO: add test data loader

    if args.overfit:
        # we train and eval with the same single training batch
        logger.info("Data in overfitting mode")
        assert not args.distributed
        train_loader = dataset.get_overfit_loader("train")
        val_loader = val_dataset.get_overfit_loader("val")
    else:
        train_loader = dataset.get_data_loader("train", distributed=args.distributed)
        val_loader = val_dataset.get_data_loader("val")
    if rank == 0:
        logger.info(f"Training loader has {len(train_loader)} batches")
        logger.info(f"Validation loader has {len(val_loader)} batches")

    # interrupts are caught and delayed for graceful termination
    def sigint_handler(signal, frame):
        logger.info("Caught keyboard interrupt signal, will terminate")
        nonlocal stop
        if stop:
            raise KeyboardInterrupt
        stop = True

    stop = False
    signal.signal(signal.SIGINT, sigint_handler)

    model = get_model(conf.model.name)(conf.model).to(device)
    if args.compile:
        model = torch.compile(model, mode=args.compile)
    loss_fn = model.loss
    if init_cp is not None:
        model.load_state_dict(init_cp["model"], strict=False)
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])
    if rank == 0 and args.print_arch:
        logger.info(f"Model: \n{model}")

    torch.backends.cudnn.benchmark = True
    if args.detect_anomaly:
        logger.info("Enabling anomaly detection")
        torch.autograd.set_detect_anomaly(True)

    optimizer_fn = {
        "sgd": torch.optim.SGD,
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "rmsprop": torch.optim.RMSprop,
    }[conf.train.optimizer]
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if conf.train.opt_regexp:
        params = filter_parameters(params, conf.train.opt_regexp)
    all_params = [p for n, p in params]
    logger.info(f"Num parameters: {sum(p.numel() for p in all_params)}")

    lr_params = pack_lr_parameters(params, conf.train.lr, conf.train.lr_scaling)
    optimizer = optimizer_fn(lr_params, lr=conf.train.lr, **conf.train.optimizer_options)
    scaler = GradScaler(enabled=args.mixed_precision is not None)
    logger.info(f"Training with mixed_precision={args.mixed_precision}")

    mp_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        None: torch.float32,  # we disable it anyway
    }[args.mixed_precision]

    results = None  # fix bug with it saving

    lr_scheduler = get_lr_scheduler(optimizer=optimizer, conf=conf.train.lr_schedule)
    logger.info(f"Using lr scheduler of type {type(lr_scheduler)}")

    if args.restore:
        optimizer.load_state_dict(init_cp["optimizer"])
        if "lr_scheduler" in init_cp:
            lr_scheduler.load_state_dict(init_cp["lr_scheduler"])

    if rank == 0:
        logger.info("Starting training with configuration:\n%s", OmegaConf.to_yaml(conf))
    losses_ = None

    def trace_handler(p):
        # torch.profiler.tensorboard_trace_handler(str(output_dir))
        output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
        print(output)
        p.export_chrome_trace("trace_" + str(p.step_num) + ".json")
        p.export_stacks("/tmp/profiler_stacks.txt", "self_cuda_time_total")

    if args.profile:
        prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=1, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(output_dir)),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        prof.__enter__()

    if conf.train.log_grad_every_iter:
        writer.watch(model, log_freq=conf.train.log_grad_every_iter)

    if conf.train.num_steps is not None:
        conf.train.epochs = conf.train.num_steps // len(train_loader) + 1
        conf.train.epochs = conf.train.epochs // (args.n_gpus if args.distributed else 1)
        logger.info(f"Setting epochs to {conf.train.epochs} to match num_steps.")

    while epoch < conf.train.epochs and not stop:
        tot_it = (len(train_loader) * epoch) * (args.n_gpus if args.distributed else 1)
        tot_n_samples = tot_it * train_loader.batch_size

        if conf.train.num_steps is not None and tot_it > conf.train.num_steps:
            logger.info(f"Reached max number of steps {conf.train.num_steps}")
            stop = True

        if rank == 0:
            logger.info(f"Starting epoch {epoch}")

        # we first run the eval
        if (
            rank == 0
            and epoch % conf.train.test_every_epoch == 0
            and (epoch > 0 or not args.no_test_0)
        ):
            for bname, eval_conf in conf.get("benchmarks", {}).items():
                logger.info(f"Running eval on {bname}")
                s, f, r = run_benchmark(
                    bname,
                    eval_conf,
                    EVAL_PATH / bname / args.experiment / str(epoch),
                    model.eval(),
                )
                for metric_name, value in s.items():
                    writer.add_scalar(f"test/{bname}/{metric_name}", value, step=tot_n_samples)
                for fig_name, fig in f.items():
                    writer.add_figure(f"figures/{bname}/{fig_name}", fig, step=tot_n_samples)

                str_results = [f"{k} {v:.3E}" for k, v in s.items() if isinstance(v, float)]
                if rank == 0:
                    logger.info(f'[Test {bname}] {{{", ".join(str_results)}}}')

        # set the seed
        set_seed(conf.train.seed + epoch)

        # update learning rate
        if conf.train.lr_schedule.on_epoch and epoch > 0:
            old_lr = optimizer.param_groups[0]["lr"]
            lr_scheduler.step(epoch)
            logger.info(f'lr changed from {old_lr} to {optimizer.param_groups[0]["lr"]}')

        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        if epoch > 0 and conf.train.dataset_callback_fn and not args.overfit:
            loaders = [train_loader]
            if conf.train.dataset_callback_on_val:
                loaders += [val_loader]
            for loader in loaders:
                if isinstance(loader.dataset, torch.utils.data.Subset):
                    getattr(loader.dataset.dataset, conf.train.dataset_callback_fn)(
                        conf.train.seed + epoch
                    )
                else:
                    getattr(loader.dataset, conf.train.dataset_callback_fn)(conf.train.seed + epoch)
        for it, data in enumerate(train_loader):
            # logger.info(f"Starting iteration {it} - epoch {epoch} - rank {rank}")
            tot_it = (len(train_loader) * epoch + it) * (args.n_gpus if args.distributed else 1)
            tot_n_samples = tot_it
            if not args.log_it:
                # We normalize the x-axis of tensorboard to num samples!
                tot_n_samples *= train_loader.batch_size

            model.train()
            optimizer.zero_grad()

            with autocast(enabled=args.mixed_precision is not None, dtype=mp_dtype):
                data = batch_to_device(data, device, non_blocking=False)
                pred = model(data)
                losses, metrics = loss_fn(pred, data, epoch, conf.train.epochs)
                loss = torch.mean(losses["total"])

            # Skip the iteration if any rank encountered a NaN
            if loss_has_nan(loss, distributed=args.distributed):
                logger.warning(f"Skipping iteration {it} due to NaN (rank {rank})")
                del pred, data, loss, losses, metrics
                torch.cuda.empty_cache()
                continue

            do_backward = loss.requires_grad
            if args.distributed:
                do_backward = torch.tensor(do_backward).float().to(device)
                torch.distributed.all_reduce(do_backward, torch.distributed.ReduceOp.PRODUCT)
                do_backward = do_backward > 0

            if do_backward:
                scaler.scale(loss).backward()
                if args.detect_anomaly:
                    # Check for params without any gradient which causes
                    # problems in distributed training with checkpointing
                    detected_anomaly = False
                    for name, param in model.named_parameters():
                        if param.grad is None and param.requires_grad:
                            logger.warning(f"param {name} has no gradient.")
                            detected_anomaly = True
                    if detected_anomaly:
                        raise RuntimeError("Detected anomaly in training.")

                if conf.train.get("clip_grad", None):
                    scaler.unscale_(optimizer)
                    try:
                        torch.nn.utils.clip_grad_norm_(
                            all_params,
                            max_norm=conf.train.clip_grad,
                            error_if_nonfinite=True,
                        )
                        scaler.step(optimizer)
                    except RuntimeError:
                        logger.warning("NaN detected in gradient clipping. Skipping iteration.")
                    scaler.update()
                else:
                    scaler.step(optimizer)
                    scaler.update()

                if not conf.train.lr_schedule.on_epoch:
                    [lr_scheduler.step() for _ in range(args.n_gpus if args.distributed else 1)]
            else:
                if rank == 0:
                    logger.warning(f"Skip iteration {it} due to detach/nan. (rank {rank})")

            if args.profile:
                prof.step()

            if it % conf.train.log_every_iter == 0:
                train_results = metrics | losses
                for k in sorted(train_results.keys()):
                    if args.distributed:
                        train_results[k] = train_results[k].sum(-1)
                        torch.distributed.reduce(train_results[k], dst=0)
                        train_results[k] /= train_loader.batch_size * args.n_gpus
                    train_results[k] = torch.mean(train_results[k], -1)
                    train_results[k] = train_results[k].item()
                if rank == 0:
                    str_losses = [f"{k} {v:.3E}" for k, v in train_results.items()]
                    logger.info(
                        "[E {} | it {}] loss {{{}}}".format(epoch, it, ", ".join(str_losses))
                    )
                    for k, v in train_results.items():
                        writer.add_scalar("training/" + k, v, tot_n_samples)

                    writer.add_scalar("training/lr", optimizer.param_groups[0]["lr"], tot_n_samples)
                    writer.add_scalar("training/epoch", epoch, tot_n_samples)

            if (
                conf.train.log_grad_every_iter is not None
                and it % conf.train.log_grad_every_iter == 0
            ):
                grad_txt = ""
                for name, param in model.named_parameters():
                    if param.grad is not None and param.requires_grad:
                        if name.endswith("bias"):
                            continue
                        writer.add_histogram(f"grad/{name}", param.grad.detach(), tot_n_samples)
                        norm = torch.norm(param.grad.detach(), 2)
                        grad_txt += f"{name} {norm.item():.3f}  \n"
                writer.add_text(f"grad/summary", grad_txt, tot_n_samples)
            del pred, data, loss, losses

            # Run validation
            if (
                (it % conf.train.eval_every_iter == 0 and (it > 0 or epoch == -int(args.no_eval_0)))
                or stop
                or it == (len(train_loader) - 1)
            ):
                with fork_rng(seed=conf.train.seed):
                    results, pr_metrics, figures = do_evaluation(
                        model,
                        val_loader,
                        device,
                        loss_fn,
                        conf.train,
                        pbar=(rank == -1),
                    )

                if rank == 0:
                    str_results = [
                        f"{k} {v:.3E}" for k, v in results.items() if isinstance(v, float)
                    ]
                    logger.info(f'[Validation] {{{", ".join(str_results)}}}')
                    for k, v in results.items():
                        if isinstance(v, dict):
                            writer.add_scalars(f"figure/val/{k}", v, tot_n_samples)
                        else:
                            writer.add_scalar("val/" + k, v, tot_n_samples)
                    for k, v in pr_metrics.items():
                        writer.add_pr_curve("val/" + k, *v, tot_n_samples)
                    # @TODO: optional always save checkpoint
                    if results[conf.train.best_key] < best_eval:
                        best_eval = results[conf.train.best_key]
                        save_experiment(
                            model,
                            optimizer,
                            lr_scheduler,
                            conf,
                            losses_,
                            results,
                            best_eval,
                            epoch,
                            tot_it,
                            output_dir,
                            stop,
                            args.distributed,
                            cp_name="checkpoint_best.tar",
                        )
                        logger.info(f"New best val: {conf.train.best_key}={best_eval}")
                    if len(figures) > 0:
                        for i, figs in enumerate(figures):
                            for name, fig in figs.items():
                                writer.add_figure(f"figures/{i}_{name}", fig, tot_n_samples)
                torch.cuda.empty_cache()  # should be cleared at the first iter

            if (tot_it % conf.train.save_every_iter == 0 and tot_it > 0) and rank == 0:
                if results is None:
                    results, _, _ = do_evaluation(
                        model,
                        val_loader,
                        device,
                        loss_fn,
                        conf.train,
                        pbar=(rank == -1),
                    )
                    best_eval = results[conf.train.best_key]
                best_eval = save_experiment(
                    model,
                    optimizer,
                    lr_scheduler,
                    conf,
                    losses_,
                    results,
                    best_eval,
                    epoch,
                    tot_it,
                    output_dir,
                    stop,
                    args.distributed,
                )

            if stop:
                break

        if rank == 0:
            best_eval = save_experiment(
                model,
                optimizer,
                lr_scheduler,
                conf,
                losses_,
                results,
                best_eval,
                epoch,
                tot_it,
                output_dir=output_dir,
                stop=stop,
                distributed=args.distributed,
            )

        epoch += 1

    logger.info(f"Finished training on process {rank}.")
    if rank == 0:
        writer.close()


def loss_has_nan(loss: torch.Tensor, distributed: bool) -> bool:
    """Check if any rank has encountered a NaN loss."""
    has_nan = torch.tensor([torch.isnan(loss).any().float()]).to(loss.device)

    # Synchronize the has_nan variable across all ranks
    if distributed:
        torch.distributed.all_reduce(has_nan, op=torch.distributed.ReduceOp.MAX)

    return has_nan.item() > 0.5


def main_worker(rank, conf, output_dir, args):
    if rank == 0:
        with capture_outputs(output_dir / "log.txt"):
            training(rank, conf, output_dir, args)
    else:
        training(rank, conf, output_dir, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str)
    parser.add_argument("--conf", type=str)
    parser.add_argument(
        "--mixed_precision",
        "--mp",
        default=None,
        type=str,
        choices=["float16", "bfloat16"],
    )
    parser.add_argument(
        "--compile",
        default=None,
        type=str,
        choices=["default", "reduce-overhead", "max-autotune"],
    )
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--restore", action="store_true")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--print_arch", "--pa", action="store_true")
    parser.add_argument("--detect_anomaly", "--da", action="store_true")
    parser.add_argument("--log_it", "--log_it", action="store_true")
    parser.add_argument("--no_eval_0", action="store_true")
    parser.add_argument("--no_test_0", action="store_true")
    parser.add_argument("dotlist", nargs="*")
    args = parser.parse_intermixed_args()

    logger.info(f"Starting experiment {args.experiment}")
    output_dir = Path(TRAINING_PATH, args.experiment)
    output_dir.mkdir(exist_ok=True, parents=True)

    conf = OmegaConf.from_cli(args.dotlist)

    if args.conf:
        initialize(version_base=None, config_path="configs")
        conf = compose(config_name=args.conf, overrides=args.dotlist)
    elif args.restore:
        restore_conf = OmegaConf.load(output_dir / "config.yaml")
        conf = OmegaConf.merge(restore_conf, conf)

    if not args.restore:
        if conf.train.seed is None:
            conf.train.seed = torch.initial_seed() & (2**32 - 1)
        OmegaConf.save(conf, str(output_dir / "config.yaml"))

    # copy geocalib and submodule into output dir
    for module in conf.train.submodules + [__module_name__]:
        mod_dir = Path(__import__(str(module)).__file__).parent
        shutil.copytree(mod_dir, output_dir / module, dirs_exist_ok=True)

    if args.distributed:
        args.n_gpus = torch.cuda.device_count()
        args.lock_file = output_dir / "distributed_lock"
        if args.lock_file.exists():
            args.lock_file.unlink()
        torch.multiprocessing.spawn(main_worker, nprocs=args.n_gpus, args=(conf, output_dir, args))
    else:
        main_worker(1, conf, output_dir, args)
