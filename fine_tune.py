# training with captions
# XXX dropped option: hypernetwork training

import argparse
import math
import os

from library.ipex_interop import init_ipex

init_ipex()

import library.train_util as train_util
import library.config_util as config_util
import library.custom_train_functions as custom_train_functions

from library.blip_caption_gui import caption_images
from library.dreambooth_folder_creation_gui import dreambooth_folder_preparation
from library.custom_logging import setup_logging
import inspect

from library.common_gui import (
    run_cmd_advanced_training,
    check_if_model_exist,
    verify_image_folder_pattern
)
from library.class_sample_images import run_cmd_sample
from library.class_command_executor import CommandExecutor
executor = CommandExecutor()

log = setup_logging()

V2_BASE_MODELS = [
    "stabilityai/stable-diffusion-2-1-base/blob/main/v2-1_512-ema-pruned",
    "stabilityai/stable-diffusion-2-1-base",
    "stabilityai/stable-diffusion-2-base",
]

# define a list of substrings to search for v_parameterization models
V_PARAMETERIZATION_MODELS = [
    "stabilityai/stable-diffusion-2-1/blob/main/v2-1_768-ema-pruned",
    "stabilityai/stable-diffusion-2-1",
    "stabilityai/stable-diffusion-2",
]

# define a list of substrings to v1.x models
V1_MODELS = [
    "CompVis/stable-diffusion-v1-4",
    "runwayml/stable-diffusion-v1-5",
]

# define a list of substrings to search for SDXL base models
SDXL_MODELS = [
    "stabilityai/stable-diffusion-xl-base-1.0",
    "stabilityai/stable-diffusion-xl-refiner-1.0",
]

ALL_PRESET_MODELS = V2_BASE_MODELS + V_PARAMETERIZATION_MODELS + V1_MODELS + SDXL_MODELS

def train_model(
    headless,
    print_only,
    pretrained_model_name_or_path,
    v2,
    v_parameterization,
    sdxl,
    logging_dir,
    train_data_dir,
    reg_data_dir,
    output_dir,
    max_resolution,
    learning_rate,
    lr_scheduler,
    lr_warmup,
    train_batch_size,
    epoch,
    save_every_n_epochs,
    mixed_precision,
    save_precision,
    seed,
    num_cpu_threads_per_process,
    cache_latents,
    cache_latents_to_disk,
    caption_extension,
    enable_bucket,
    gradient_checkpointing,
    fp8_base,
    full_fp16,
    # no_token_padding,
    stop_text_encoder_training_pct,
    min_bucket_reso,
    max_bucket_reso,
    # use_8bit_adam,
    xformers,
    save_model_as,
    shuffle_caption,
    save_state,
    resume,
    prior_loss_weight,
    text_encoder_lr,
    unet_lr,
    network_dim,
    lora_network_weights,
    dim_from_weights,
    color_aug,
    flip_aug,
    clip_skip,
    num_processes,
    num_machines,
    multi_gpu,
    gpu_ids,
    gradient_accumulation_steps,
    mem_eff_attn,
    output_name,
    model_list,  # Keep this. Yes, it is unused here but required given the common list used
    max_token_length,
    max_train_epochs,
    max_train_steps,
    max_data_loader_n_workers,
    network_alpha,
    training_comment,
    keep_tokens,
    lr_scheduler_num_cycles,
    lr_scheduler_power,
    persistent_data_loader_workers,
    bucket_no_upscale,
    random_crop,
    bucket_reso_steps,
    v_pred_like_loss,
    caption_dropout_every_n_epochs,
    caption_dropout_rate,
    optimizer,
    optimizer_args,
    lr_scheduler_args,
    max_grad_norm,
    noise_offset_type,
    noise_offset,
    adaptive_noise_scale,
    multires_noise_iterations,
    multires_noise_discount,
    LoRA_type,
    factor,
    use_cp,
    use_tucker,
    use_scalar,
    rank_dropout_scale,
    constrain,
    rescaled,
    train_norm,
    decompose_both,
    train_on_input,
    conv_dim,
    conv_alpha,
    sample_every_n_steps,
    sample_every_n_epochs,
    sample_sampler,
    sample_prompts,
    additional_parameters,
    vae_batch_size,
    min_snr_gamma,
    down_lr_weight,
    mid_lr_weight,
    up_lr_weight,
    block_lr_zero_threshold,
    block_dims,
    block_alphas,
    conv_block_dims,
    conv_block_alphas,
    weighted_captions,
    unit,
    save_every_n_steps,
    save_last_n_steps,
    save_last_n_steps_state,
    use_wandb,
    wandb_api_key,
    scale_v_pred_loss_like_noise_pred,
    scale_weight_norms,
    network_dropout,
    rank_dropout,
    module_dropout,
    sdxl_cache_text_encoder_outputs,
    sdxl_no_half_vae,
    full_bf16,
    min_timestep,
    max_timestep,
    vae,
    LyCORIS_preset,
    debiased_estimation_loss,
):
    # Get list of function parameters and values
    parameters = list(locals().items())
    global command_running

    print_only_bool = True if print_only.get("label") == "True" else False
    log.info(f"Start training LoRA {LoRA_type} ...")
    headless_bool = True if headless.get("label") == "True" else False

    if pretrained_model_name_or_path == "":
        msg="Source model information is missing"
        print(msg)
        return

    if train_data_dir == "":
        msg="Image folder path is missing"
        print(msg)
        return

    # Check if there are files with the same filename but different image extension... warn the user if it is the case.

    if not os.path.exists(train_data_dir):
        print('Image Folder does not exist')
        print(train_data_dir)
        return

    if not verify_image_folder_pattern(train_data_dir):
        return

    if reg_data_dir != "":
        if not os.path.exists(reg_data_dir):
            print('REg folder does not exist')
            print(reg_data_dir)
            return

        if not verify_image_folder_pattern(reg_data_dir):
            return

    if output_dir == "":
        print('output folder path is missing')
        return

    if int(bucket_reso_steps) < 1:
        msg="Bucket resolution steps need to be greater than 0",
        print(msg)
        return

    if noise_offset == "":
        noise_offset = 0

    if float(noise_offset) > 1 or float(noise_offset) < 0:
        msg="Noise offset need to be a value between 0 and 1"
        print(msg)
        return

    # if float(noise_offset) > 0 and (
    #     multires_noise_iterations > 0 or multires_noise_discount > 0
    # ):
    #     output_message(
    #         msg="noise offset and multires_noise can't be set at the same time. Only use one or the other.",
    #         title='Error',
    #         headless=headless_bool,
    #     )
    #     return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if stop_text_encoder_training_pct > 0:
        msg='Output "stop text encoder training" is not yet supported. Ignoring',
        print(msg)
        stop_text_encoder_training_pct = 0

    if check_if_model_exist(
        output_name, output_dir, save_model_as, headless=headless_bool
    ):
        return

    # if optimizer == 'Adafactor' and lr_warmup != '0':
    #     output_message(
    #         msg="Warning: lr_scheduler is set to 'Adafactor', so 'LR warmup (% of steps)' will be considered 0.",
    #         title='Warning',
    #         headless=headless_bool,
    #     )
    #     lr_warmup = '0'

    # If string is empty set string to 0.
    if text_encoder_lr == "":
        text_encoder_lr = 0
    if unet_lr == "":
        unet_lr = 0

    # Get a list of all subfolders in train_data_dir
    subfolders = [
        f
        for f in os.listdir(train_data_dir)
        if os.path.isdir(os.path.join(train_data_dir, f))
    ]

    total_steps = 0

    # Loop through each subfolder and extract the number of repeats
    for folder in subfolders:
        try:
            # Extract the number of repeats from the folder name
            repeats = int(folder.split("_")[0])

            # Count the number of images in the folder
            num_images = len(
                [
                    f
                    for f, lower_f in (
                        (file, file.lower())
                        for file in os.listdir(os.path.join(train_data_dir, folder))
                    )
                    if lower_f.endswith((".jpg", ".jpeg", ".png", ".webp"))
                ]
            )

            log.info(f"Folder {folder}: {num_images} images found")

            # Calculate the total number of steps for this folder
            steps = repeats * num_images

            # log.info the result
            log.info(f"Folder {folder}: {steps} steps")

            total_steps += steps

        except ValueError:
            # Handle the case where the folder name does not contain an underscore
            log.info(f"Error: '{folder}' does not contain an underscore, skipping...")

    if reg_data_dir == "":
        reg_factor = 1
    else:
        log.warning(
            "Regularisation images are used... Will double the number of steps required..."
        )
        reg_factor = 2

    log.info(f"Total steps: {total_steps}")
    log.info(f"Train batch size: {train_batch_size}")
    log.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    log.info(f"Epoch: {epoch}")
    log.info(f"Regulatization factor: {reg_factor}")

    if max_train_steps == "" or max_train_steps == "0":
        # calculate max_train_steps
        max_train_steps = int(
            math.ceil(
                float(total_steps)
                / int(train_batch_size)
                / int(gradient_accumulation_steps)
                * int(epoch)
                * int(reg_factor)
            )
        )
        log.info(
            f"max_train_steps ({total_steps} / {train_batch_size} / {gradient_accumulation_steps} * {epoch} * {reg_factor}) = {max_train_steps}"
        )

    # calculate stop encoder training
    if stop_text_encoder_training_pct == None:
        stop_text_encoder_training = 0
    else:
        stop_text_encoder_training = math.ceil(
            float(max_train_steps) / 100 * int(stop_text_encoder_training_pct)
        )
    log.info(f"stop_text_encoder_training = {stop_text_encoder_training}")

    lr_warmup_steps = round(float(int(lr_warmup) * int(max_train_steps) / 100))
    log.info(f"lr_warmup_steps = {lr_warmup_steps}")

    run_cmd = "accelerate launch"

    run_cmd += run_cmd_advanced_training(
        num_processes=num_processes,
        num_machines=num_machines,
        multi_gpu=multi_gpu,
        gpu_ids=gpu_ids,
        num_cpu_threads_per_process=num_cpu_threads_per_process,
    )

    if sdxl:
        run_cmd += f' "./sdxl_train_network.py"'
    else:
        run_cmd += f' "./train_network.py"'

    if LoRA_type == "LyCORIS/Diag-OFT":
        network_module = "lycoris.kohya"
        network_args = f' "preset={LyCORIS_preset}" "conv_dim={conv_dim}" "conv_alpha={conv_alpha}" "module_dropout={module_dropout}" "use_tucker={use_tucker}" "use_scalar={use_scalar}" "rank_dropout_scale={rank_dropout_scale}" "constrain={constrain}" "rescaled={rescaled}" "algo=diag-oft" '

    if LoRA_type == "LyCORIS/DyLoRA":
        network_module = "lycoris.kohya"
        network_args = f' "preset={LyCORIS_preset}" "conv_dim={conv_dim}" "conv_alpha={conv_alpha}" "use_tucker={use_tucker}" "block_size={unit}" "rank_dropout={rank_dropout}" "module_dropout={module_dropout}" "algo=dylora" "train_norm={train_norm}"'

    if LoRA_type == "LyCORIS/GLoRA":
        network_module = "lycoris.kohya"
        network_args = f' "preset={LyCORIS_preset}" "conv_dim={conv_dim}" "conv_alpha={conv_alpha}" "rank_dropout={rank_dropout}" "module_dropout={module_dropout}" "rank_dropout_scale={rank_dropout_scale}" "algo=glora" "train_norm={train_norm}"'

    if LoRA_type == "LyCORIS/iA3":
        network_module = "lycoris.kohya"
        network_args = f' "preset={LyCORIS_preset}" "conv_dim={conv_dim}" "conv_alpha={conv_alpha}" "train_on_input={train_on_input}" "algo=ia3"'

    if LoRA_type == "LoCon" or LoRA_type == "LyCORIS/LoCon":
        network_module = "lycoris.kohya"
        network_args = f' "preset={LyCORIS_preset}" "conv_dim={conv_dim}" "conv_alpha={conv_alpha}" "rank_dropout={rank_dropout}" "module_dropout={module_dropout}" "use_tucker={use_tucker}" "use_scalar={use_scalar}" "rank_dropout_scale={rank_dropout_scale}" "algo=locon" "train_norm={train_norm}"'

    if LoRA_type == "LyCORIS/LoHa":
        network_module = "lycoris.kohya"
        network_args = f' "preset={LyCORIS_preset}" "conv_dim={conv_dim}" "conv_alpha={conv_alpha}" "rank_dropout={rank_dropout}" "module_dropout={module_dropout}" "use_tucker={use_tucker}" "use_scalar={use_scalar}" "rank_dropout_scale={rank_dropout_scale}" "algo=loha" "train_norm={train_norm}"'

    if LoRA_type == "LyCORIS/LoKr":
        network_module = "lycoris.kohya"
        network_args = f' "preset={LyCORIS_preset}" "conv_dim={conv_dim}" "conv_alpha={conv_alpha}" "rank_dropout={rank_dropout}" "module_dropout={module_dropout}" "factor={factor}" "use_cp={use_cp}" "use_scalar={use_scalar}" "decompose_both={decompose_both}" "rank_dropout_scale={rank_dropout_scale}" "algo=lokr" "train_norm={train_norm}"'

    if LoRA_type == "LyCORIS/Native Fine-Tuning":
        network_module = "lycoris.kohya"
        network_args = f' "preset={LyCORIS_preset}" "rank_dropout={rank_dropout}" "module_dropout={module_dropout}" "use_tucker={use_tucker}" "use_scalar={use_scalar}" "rank_dropout_scale={rank_dropout_scale}" "algo=full" "train_norm={train_norm}"'

    if LoRA_type in ["Kohya LoCon", "Standard"]:
        kohya_lora_var_list = [
            "down_lr_weight",
            "mid_lr_weight",
            "up_lr_weight",
            "block_lr_zero_threshold",
            "block_dims",
            "block_alphas",
            "conv_block_dims",
            "conv_block_alphas",
            "rank_dropout",
            "module_dropout",
        ]

        network_module = "networks.lora"
        kohya_lora_vars = {
            key: value
            for key, value in vars().items()
            if key in kohya_lora_var_list and value
        }

        network_args = ""
        if LoRA_type == "Kohya LoCon":
            network_args += f' conv_dim="{conv_dim}" conv_alpha="{conv_alpha}"'

        for key, value in kohya_lora_vars.items():
            if value:
                network_args += f' {key}="{value}"'

    if LoRA_type in [
        "LoRA-FA",
    ]:
        kohya_lora_var_list = [
            "down_lr_weight",
            "mid_lr_weight",
            "up_lr_weight",
            "block_lr_zero_threshold",
            "block_dims",
            "block_alphas",
            "conv_block_dims",
            "conv_block_alphas",
            "rank_dropout",
            "module_dropout",
        ]

        network_module = "networks.lora_fa"
        kohya_lora_vars = {
            key: value
            for key, value in vars().items()
            if key in kohya_lora_var_list and value
        }

        network_args = ""
        if LoRA_type == "Kohya LoCon":
            network_args += f' conv_dim="{conv_dim}" conv_alpha="{conv_alpha}"'

        for key, value in kohya_lora_vars.items():
            if value:
                network_args += f' {key}="{value}"'

    if LoRA_type in ["Kohya DyLoRA"]:
        kohya_lora_var_list = [
            "conv_dim",
            "conv_alpha",
            "down_lr_weight",
            "mid_lr_weight",
            "up_lr_weight",
            "block_lr_zero_threshold",
            "block_dims",
            "block_alphas",
            "conv_block_dims",
            "conv_block_alphas",
            "rank_dropout",
            "module_dropout",
            "unit",
        ]

        network_module = "networks.dylora"
        kohya_lora_vars = {
            key: value
            for key, value in vars().items()
            if key in kohya_lora_var_list and value
        }

        network_args = ""

        for key, value in kohya_lora_vars.items():
            if value:
                network_args += f' {key}="{value}"'

    network_train_text_encoder_only = False
    network_train_unet_only = False

    # Convert learning rates to float once and store the result for re-use
    if text_encoder_lr is None:
        msg="Please input valid Text Encoder learning rate (between 0 and 1)"
        print(msg)
        return
    if unet_lr is None:
        msg="Please input valid Unet learning rate (between 0 and 1)"
        print(msg)
        return
    text_encoder_lr_float = float(text_encoder_lr)
    unet_lr_float = float(unet_lr)
    
    

    # Determine the training configuration based on learning rate values
    if text_encoder_lr_float == 0 and unet_lr_float == 0:
        if float(learning_rate) == 0:
            msg="Please input learning rate values."
            print(msg)
            return
    elif text_encoder_lr_float != 0 and unet_lr_float == 0:
        network_train_text_encoder_only = True
    elif text_encoder_lr_float == 0 and unet_lr_float != 0:
        network_train_unet_only = True
    # If both learning rates are non-zero, no specific flags need to be set

    run_cmd += run_cmd_advanced_training(
        adaptive_noise_scale=adaptive_noise_scale,
        additional_parameters=additional_parameters,
        bucket_no_upscale=bucket_no_upscale,
        bucket_reso_steps=bucket_reso_steps,
        cache_latents=cache_latents,
        cache_latents_to_disk=cache_latents_to_disk,
        cache_text_encoder_outputs=True if sdxl and sdxl_cache_text_encoder_outputs else None,
        caption_dropout_every_n_epochs=caption_dropout_every_n_epochs,
        caption_dropout_rate=caption_dropout_rate,
        caption_extension=caption_extension,
        clip_skip=clip_skip,
        color_aug=color_aug,
        debiased_estimation_loss=debiased_estimation_loss,
        dim_from_weights=dim_from_weights,
        enable_bucket=enable_bucket,
        epoch=epoch,
        flip_aug=flip_aug,
        fp8_base=fp8_base,
        full_bf16=full_bf16,
        full_fp16=full_fp16,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        keep_tokens=keep_tokens,
        learning_rate=learning_rate,
        logging_dir=logging_dir,
        lora_network_weights=lora_network_weights,
        lr_scheduler=lr_scheduler,
        lr_scheduler_args=lr_scheduler_args,
        lr_scheduler_num_cycles=lr_scheduler_num_cycles,
        lr_scheduler_power=lr_scheduler_power,
        lr_warmup_steps=lr_warmup_steps,
        max_bucket_reso=max_bucket_reso,
        max_data_loader_n_workers=max_data_loader_n_workers,
        max_grad_norm=max_grad_norm,
        max_resolution=max_resolution,
        max_timestep=max_timestep,
        max_token_length=max_token_length,
        max_train_epochs=max_train_epochs,
        max_train_steps=max_train_steps,
        mem_eff_attn=mem_eff_attn,
        min_bucket_reso=min_bucket_reso,
        min_snr_gamma=min_snr_gamma,
        min_timestep=min_timestep,
        mixed_precision=mixed_precision,
        multires_noise_discount=multires_noise_discount,
        multires_noise_iterations=multires_noise_iterations,
        network_alpha=network_alpha,
        network_args=network_args,
        network_dim=network_dim,
        network_dropout=network_dropout,
        network_module=network_module,
        network_train_unet_only=network_train_unet_only,
        network_train_text_encoder_only=network_train_text_encoder_only,
        no_half_vae=True if sdxl and sdxl_no_half_vae else None,
        # no_token_padding=no_token_padding,
        noise_offset=noise_offset,
        noise_offset_type=noise_offset_type,
        optimizer=optimizer,
        optimizer_args=optimizer_args,
        output_dir=output_dir,
        output_name=output_name,
        persistent_data_loader_workers=persistent_data_loader_workers,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        prior_loss_weight=prior_loss_weight,
        random_crop=random_crop,
        reg_data_dir=reg_data_dir,
        resume=resume,
        save_every_n_epochs=save_every_n_epochs,
        save_every_n_steps=save_every_n_steps,
        save_last_n_steps=save_last_n_steps,
        save_last_n_steps_state=save_last_n_steps_state,
        save_model_as=save_model_as,
        save_precision=save_precision,
        save_state=save_state,
        scale_v_pred_loss_like_noise_pred=scale_v_pred_loss_like_noise_pred,
        scale_weight_norms=scale_weight_norms,
        seed=seed,
        shuffle_caption=shuffle_caption,
        stop_text_encoder_training=stop_text_encoder_training,
        text_encoder_lr=text_encoder_lr,
        train_batch_size=train_batch_size,
        train_data_dir=train_data_dir,
        training_comment=training_comment,
        unet_lr=unet_lr,
        use_wandb=use_wandb,
        v2=v2,
        v_parameterization=v_parameterization,
        v_pred_like_loss=v_pred_like_loss,
        vae=vae,
        vae_batch_size=vae_batch_size,
        wandb_api_key=wandb_api_key,
        weighted_captions=weighted_captions,
        xformers=xformers,
    )

    run_cmd += run_cmd_sample(
        sample_every_n_steps,
        sample_every_n_epochs,
        sample_sampler,
        sample_prompts,
        output_dir,
    )

    if print_only_bool:
        log.warning(
            "Here is the trainer command as a reference. It will not be executed:\n"
        )
        print(run_cmd)
    else:
       

        log.info(run_cmd)
        # Run the command
        executor.execute_command(run_cmd=run_cmd)

        # # check if output_dir/last is a folder... therefore it is a diffuser model
        # last_dir = pathlib.Path(f'{output_dir}/{output_name}')

        # if not last_dir.is_dir():
        #     # Copy inference model for v2 if required
        #     save_inference_file(
        #         output_dir, v2, v_parameterization, output_name
        #     )
# define a list of substrings to search for v2 base models

def update_my_data(my_data):
    # Update the optimizer based on the use_8bit_adam flag
    use_8bit_adam = my_data.get("use_8bit_adam", False)
    my_data.setdefault("optimizer", "AdamW8bit" if use_8bit_adam else "AdamW")

    # Update model_list to custom if empty or pretrained_model_name_or_path is not a preset model
    model_list = my_data.get("model_list", [])
    pretrained_model_name_or_path = my_data.get("pretrained_model_name_or_path", "")
    if not model_list or pretrained_model_name_or_path not in ALL_PRESET_MODELS:
        my_data["model_list"] = "custom"

    # Convert values to int if they are strings
    for key in ["epoch", "save_every_n_epochs", "lr_warmup"]:
        value = my_data.get(key, 0)
        if isinstance(value, str) and value.strip().isdigit():
            my_data[key] = int(value)
        elif not value:
            my_data[key] = 0

    # Convert values to float if they are strings
    for key in ["noise_offset", "learning_rate", "text_encoder_lr", "unet_lr"]:
        value = my_data.get(key, 0)
        if isinstance(value, str) and value.strip().isdigit():
            my_data[key] = float(value)
        elif not value:
            my_data[key] = 0

    # Update LoRA_type if it is set to LoCon
    if my_data.get("LoRA_type", "Standard") == "LoCon":
        my_data["LoRA_type"] = "LyCORIS/LoCon"

    # Update model save choices due to changes for LoRA and TI training
    if "save_model_as" in my_data:
        if (
            my_data.get("LoRA_type") or my_data.get("num_vectors_per_token")
        ) and my_data.get("save_model_as") not in ["safetensors", "ckpt"]:
            message = "Updating save_model_as to safetensors because the current value in the config file is no longer applicable to {}"
            if my_data.get("LoRA_type"):
                log.info(message.format("LoRA"))
            if my_data.get("num_vectors_per_token"):
                log.info(message.format("TI"))
            my_data["save_model_as"] = "safetensors"

    # Update xformers if it is set to True and is a boolean
    xformers_value = my_data.get("xformers", None)
    if isinstance(xformers_value, bool):
        if xformers_value:
            my_data["xformers"] = "xformers"
        else:
            my_data["xformers"] = "none"

    return my_data

def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    train_util.add_sd_models_arguments(parser)
    train_util.add_dataset_arguments(parser, False, True, True)
    train_util.add_training_arguments(parser, False)
    train_util.add_sd_saving_arguments(parser)
    train_util.add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)
    custom_train_functions.add_custom_train_arguments(parser)

    parser.add_argument("--diffusers_xformers", action="store_true", help="use xformers by diffusers / Diffusersでxformersを使用する")
    parser.add_argument("--train_text_encoder", action="store_true", help="train text encoder / text encoderも学習する")
    parser.add_argument(
        "--learning_rate_te",
        type=float,
        default=None,
        help="learning rate for text encoder, default is same as unet / Text Encoderの学習率、デフォルトはunetと同じ",
    )

    return parser

if __name__ == "__main__":
    import json
    parser = setup_parser()
    parser.add_argument('--remote_training_data_dir')
    parser.add_argument('--remote_instance_keyword')
    parser.add_argument('--remote_training_repeats')
    parser.add_argument('--remote_class_keyword')
    parser.add_argument('--remote_output_dir')
    args = parser.parse_args()

    config_path = args.config_file
    configs = json.load(open(config_path))

    # TODO get female reg
    assert args.remote_class_keyword == 'man' or args.remote_class_keyword == 'woman'
    default_reg_dir = f'/workspace/kohya_ss/dataset/reg/{args.remote_class_keyword}'

    #BLIP ARGS
    caption_file_ext = '.txt'
    batch_size = 8
    num_beams = 1
    top_p = 0.9
    max_length = 50
    min_length = 10
    beam_search = True
    prefix = args.remote_instance_keyword
    postfix = ''
    
    print('captioning...')
    caption_images(
        args.remote_training_data_dir,
        caption_file_ext,
        batch_size,
        num_beams,
        top_p,
        max_length,
        min_length,
        beam_search,
        prefix,
        postfix
    )

    print('prepping dataset...')
    updated_args = dreambooth_folder_preparation(
        args.remote_training_data_dir,
        args.remote_training_repeats,
        args.remote_instance_keyword,
        default_reg_dir,
        1,
        args.remote_class_keyword,
        args.remote_output_dir
    )

    #- "update trin_data_dir" AND "reg_data_dir" AND "output_dir"
    d = json.load(open(args.config_file))
    for k,v in updated_args.items():
        if k in d: 
            d[k] = v
    # remove train config args
    d.pop('train_config', None)
    with open(args.config_file, 'w') as f:
        json.dump(d,f)

    args = train_util.read_config_from_file(args, parser)

    #train(args)
    # TODO it takes like 20 secs to boot up the gui, just call accelerate directly? 
    my_data = update_my_data(d)

    train_model_args = inspect.getfullargspec(train_model).args

    filtered_args = {k:my_data[k] for k in train_model_args if k in my_data}

    headless = {'label': 'True'}
    print_only = {'label': 'True'}
 
    train_model(headless, print_only, **filtered_args)
    # sample cmd:
    '''
    accelerate launch --num_cpu_threads_per_process=2 "./sdxl_train_network.py" --bucket_no_upscale --bucket_reso_steps=64 --cache_latents --cache_latents_to_disk --enable_bucket            
                         --min_bucket_reso=256 --max_bucket_reso=2048 --gradient_checkpointing --learning_rate="1.0" --lr_scheduler="constant" --lr_scheduler_num_cycles="5" --max_data_loader_n_workers="0"       
                         --max_grad_norm="1" --resolution="1024,1024" --max_train_steps="5700" --mixed_precision="bf16" --network_alpha="64" --network_dim=256 --network_module=networks.lora --no_half_vae        
                         --optimizer_type="DAdaptation" --output_dir="/workspace/kohya_ss/dataset/minjunes/minjunes_02-28-13:39" --output_name="bwell2-sdxl-dadaption"                                             
                         --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" --reg_data_dir="/workspace/kohya_ss/dataset/minjunes/minjunes_02-28-13:39/reg" --save_every_n_epochs="1"       
                         --save_model_as=safetensors --save_precision="bf16" --text_encoder_lr=1.0 --train_batch_size="1" --train_data_dir="/workspace/kohya_ss/dataset/minjunes/minjunes_02-28-13:39/img"         
                         --unet_lr=1.0 --xformers   
    '''
