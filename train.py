import torch
import time
import os
from functools import partial
from utils import(
    set_training,
    get_lr_util,
    save_model_util,
    cross_entropy_loss,
    load_json, hf_permission,
    check_bfloat16_support
)
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from model import ASLM, make_peft_model
from data import DataLoaderLite



init_process_group(backend='nccl')
ddp_rank = int(os.environ['RANK'])
ddp_local_rank = int(os.environ['LOCAL_RANK'])
world_size = int(os.environ['WORLD_SIZE'])
device = f'cuda:{ddp_local_rank}'
torch.cuda.set_device(device)


config = load_json("params.json", as_holder=True)
hf_permission(config.hf_token_id)


save_model = partial(
    save_model_util,
    repo_name=config.hf_repo_name,
    user_name=config.hf_user_name,
)


get_lr = partial(
    get_lr_util,
    max_steps=config.num_iters,
    warmup_steps=config.warmup_steps,
)


LOG_DIR =  config.log_dir
LOG_FILE = config.log_filename
SAVE_FREQ = config.save_freq


dtype = torch.bfloat16 if check_bfloat16_support(ddp_local_rank == 0) else torch.float16



def training_loop(
    model,
    train_iter,
    val_iter,
    opt,
    num_iters,
    val_nums=500,
    val_loss_steps=20,
    scaler=None,
    max_norm=1.0,
    accum_steps=1,
    rank=0,
    compute_dtype=torch.float16,
    
):
    
    dev = torch.cuda.device(rank)
    is_float16 = compute_dtype is torch.float16
    is_master_process = rank == 0
    for step in range(num_iters):
        loss_accum = 0.0
        t0 = time.time()
        set_training(model)
        opt.zero_grad()
        for grad_step in range(accum_steps):
          inputs, labels = train_iter.next_batch()
          inputs = inputs.to(dev)
          labels = labels.to(dev)
          model.require_backward_grad_sync = (grad_step == accum_steps - 1)  
          with torch.autocast(device_type='cuda', dtype=compute_dtype):
              preds = model(inputs).logits
              loss = cross_entropy_loss(preds, labels)
          
          loss = loss / accum_steps
          loss_accum+= loss.detach()
          if is_float16:
             scaler.scale(loss).backward()
          else:
              loss.backward() 

        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        norm = 0.0
        if max_norm is not None:
          if is_float16:
              scaler.unscale_(opt)
          norm = torch.nn.utils.clip_grad_norm(model.parameters(), max_norm)  

        lr = get_lr(step)
        for param_group in opt.param_groups:
            param_group['lr'] = lr    
        if is_float16:
           scaler.step(opt)
           scaler.update()
        else:
          opt.step()   
        torch.cuda.synchronize()
        opt.zero_grad()

        last_step = (step == num_iters - 1)
        # once in a while evaluate our validation loss
        if step % val_nums == 0 or last_step:
            set_training(model, False)
           # val_iter.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                for _ in range(val_loss_steps):
                    x, y = val_iter.next_batch()
                    x, y = x.to(dev), y.to(dev)
                    with torch.autocast(device_type="cuda", dtype=compute_dtype):
                        logits = model(x).logits
                        loss = cross_entropy_loss(logits, y)

                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()
          
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if is_master_process:
                print(f"validation loss: {val_loss_accum.item():.4f}")
                with open(LOG_FILE, "a") as f:
                    f.write(f"{step} val {val_loss_accum.item():.4f}\n")
                if step > 0 and (step % SAVE_FREQ == 0 or last_step):
                    # optionally write model checkpoints
                    checkpoint_path = os.path.join(LOG_DIR)
                    save_model(model.module, checkpoint_path, push_to_hf=True)

        t1 = time.time()
        dt = t1 - t0 
        tokens_processed = train_iter.B * train_iter.T * accum_steps * world_size
        tokens_per_sec = tokens_processed / dt
        if is_master_process:
            print(f"step {step:5d} | loss: {loss_accum.item():.6f} | norm:{norm:.3f}|lr {lr:.4e} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
            with open(LOG_FILE, "a") as f:
                f.write(f"{step} train {loss_accum.item():.6f}\n")


model = ASLM(config.model_id, ddp_local_rank==0)
model = make_peft_model(model,ddp_local_rank==0)
model = model.to(device)
Opt = torch.optim.AdamW([p  for p in model.parameters() if p.requires_grad], 1e-4)
model = DDP(model)


train_loader = DataLoaderLite(
    config.data_dir,
    config.batch_size,
    config.seq_len,
    ddp_local_rank,
    world_size,
    split="train"
)

val_loader = DataLoaderLite(
    config.data_dir,
    config.batch_size,
    config.seq_len,
    ddp_local_rank,
    world_size,
    split="val"
)

training_loop(
    model,
    train_loader,
    val_loader,
    Opt,
    config.num_iters,
    config.val_steps,
    accum_steps=config.accum_steps,
    compute_dtype=dtype
)

destroy_process_group()
