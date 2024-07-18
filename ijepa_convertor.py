import torch
import re

def convert_vit(src_state_dict, tgt_format="monolith", prefix=""):
    mapping_mono_src = {
        "encoder.transformer_encoder.layers": "module.blocks",
        "ffn.ffn.0": "mlp.fc1",
        "ffn.ffn.1": "mlp.fc2",
        "embedding_layer.linear_proj": "patch_embed.proj",
        "embedding_layer.position_embeddings": "pos_embed",
    }

    mapping_src_mono = {v:k for k, v in mapping_mono_src.items()}

    mapping = mapping_mono_src if tgt_format == "src" else mapping_src_mono

    target_state_dict = {}
    for k, val in src_state_dict.items():
        out_key = k
        if "attn" in k:
            continue
        for src_key, tgt_key in mapping.items():
            out_key = out_key.replace(src_key, tgt_key)

        out_key = prefix + out_key
        if "embedding_layer.position_embeddings" in out_key or "pos_embed" in out_key:
            if tgt_format == "src":
                target_state_dict[out_key] = val.unsqueeze(0)
            elif tgt_format == "monolith":
                target_state_dict[out_key] = val.squeeze()
        else:
            target_state_dict[out_key] = val
        
        print(f"matched {k}: {out_key}")
    
    if tgt_format == "monolith":
        for k, val in src_state_dict.items():
            if "attn.qkv" in k:
                out_key = k.replace("module.blocks", "encoder.transformer_encoder.layers")
                layer_num = re.findall("\d+", k)[0]
                weight = torch.split(val, val.shape[0]//3, dim=0)
                for i, _suffix in enumerate(["self_attn.proj_q_dense_layer", "self_attn.proj_k_dense_layer", "self_attn.proj_v_dense_layer"]):
                    subkey = out_key.replace("attn.qkv", _suffix)
                    target_state_dict[prefix + subkey] = weight[i]

                    print(f"matched {k}: {prefix + subkey}")
            elif "attn.proj" in k:
                out_key = k.replace("module.blocks", "encoder.transformer_encoder.layers")
                out_key = out_key.replace("attn.proj", "self_attn.proj_output_dense_layer")
                print(f"matched {k}: {prefix + out_key}")
                target_state_dict[prefix + out_key] = val

    elif tgt_format == "src":
        for k, val in src_state_dict.items():
            if "self_attn.proj_q_dense_layer" in k:
                out_key = k.replace("encoder.transformer_encoder.layers", "module.blocks")
                layer_num = re.findall("\d+", k)[0]
                weight_k = src_state_dict[k.replace("proj_q_dense_layer", "proj_k_dense_layer")]
                weight_v = src_state_dict[k.replace("proj_q_dense_layer", "proj_v_dense_layer")]

                out_key = out_key.replace("self_attn.proj_q_dense_layer", "attn.qkv")
                target_state_dict[prefix + out_key] = torch.cat([val, weight_k, weight_v], dim=0)
                print(f"matched {k}: {prefix + out_key}")
            elif "self_attn.proj_output_dense_layer" in k:
                out_key = k.replace("encoder.transformer_encoder.layers", "module.blocks")
                out_key = out_key.replace("self_attn.proj_output_dense_layer", "attn.proj")
                print(f"matched {k}: {prefix + out_key}")
                target_state_dict[prefix + out_key] = val

    
    return target_state_dict
    

def convert_predictor(src_state_dict, tgt_format="monolith", prefix=""):
    mapping_mono_src = {
        "encoder.transformer_encoder.layers": "module.predictor_blocks",
        "ffn.ffn.0": "mlp.fc1",
        "ffn.ffn.1": "mlp.fc2",
        "embedding_layer.linear_proj": "patch_embed.proj",
        "embedding_layer.position_embeddings": "pos_embed"
    }

    mapping_src_mono = {v:k for k, v in mapping_mono_src.items()}

    mapping = mapping_mono_src if tgt_format == "src" else mapping_src_mono

    target_state_dict = {}
    for k, val in src_state_dict.items():
        out_key = k
        if "attn" in k:
            continue
        for src_key, tgt_key in mapping.items():
            out_key = out_key.replace(src_key, tgt_key)

        out_key = prefix + out_key
        if "embedding_layer.position_embeddings" in out_key or "pos_embed" in out_key:
            if tgt_format == "src":
                target_state_dict[out_key] = val.unsqueeze(0)
            elif tgt_format == "monolith":
                target_state_dict[out_key] = val.squeeze()
        else:
            target_state_dict[out_key] = val
        
        print(f"matched {k}: {out_key}")
    
    if tgt_format == "monolith":
        for k, val in src_state_dict.items():
            if "attn.qkv" in k:
                out_key = k.replace("module.predictor_blocks", "encoder.transformer_encoder.layers")
                layer_num = re.findall("\d+", k)[0]
                weight = torch.split(val, val.shape[0]//3, dim=0)
                for i, _suffix in enumerate(["self_attn.proj_q_dense_layer", "self_attn.proj_k_dense_layer", "self_attn.proj_v_dense_layer"]):
                    subkey = out_key.replace("attn.qkv", _suffix)
                    target_state_dict[prefix + subkey] = weight[i]
                    print(f"matched {k}: {prefix + subkey}")
            elif "attn.proj" in k:
                out_key = k.replace("module.predictor_blocks", "encoder.transformer_encoder.layers")
                out_key = out_key.replace("attn.proj", "self_attn.proj_output_dense_layer")
                print(f"matched {k}: {prefix + out_key}")
                target_state_dict[prefix + out_key] = val
    elif tgt_format == "src":
        for k, val in src_state_dict.items():
            if "self_attn.proj_q_dense_layer" in k:
                out_key = k.replace("encoder.transformer_encoder.layers", "module.predictor_blocks")
                layer_num = re.findall("\d+", k)[0]
                weight_k = src_state_dict[k.replace("proj_q_dense_layer", "proj_k_dense_layer")]
                weight_v = src_state_dict[k.replace("proj_q_dense_layer", "proj_v_dense_layer")]

                out_key = out_key.replace("self_attn.proj_q_dense_layer", "attn.qkv") 
                target_state_dict[prefix + out_key] = torch.cat([val, weight_k, weight_v], dim=0)
                print(f"matched {k}: {prefix + out_key}")
            elif "self_attn.proj_output_dense_layer" in k:
                out_key = k.replace("encoder.transformer_encoder.layers", "module.predictor_blocks")
                out_key = out_key.replace("self_attn.proj_output_dense_layer", "attn.proj")
                print(f"matched {k}: {prefix + out_key}")
                target_state_dict[prefix + out_key] = val
    
    return target_state_dict


def convert_ijepa_checkpoint(src_ckpt_path, convert_to="monolith"):
    src_state_dict = torch.load(src_ckpt_path, map_location=torch.device("cpu"))

    if convert_to == "monolith":
        converted_encoder = convert_vit(src_state_dict["encoder"], tgt_format="monolith", prefix="image_model_trunks.model.0.")
        converted_target_encoder = convert_vit(src_state_dict["target_encoder"], tgt_format="monolith", prefix="image_model_trunks.model.1.")
        converted_predictor = convert_predictor(src_state_dict["predictor"], tgt_format="monolith", prefix="heads.model.0.")

        out_state_dict = {}
        out_state_dict.update(converted_encoder)
        out_state_dict.update(converted_target_encoder)
        out_state_dict.update(converted_predictor)
        state_dict = {"model": out_state_dict}

    elif convert_to == "src":
        converted_encoder = convert_vit({k.replace("image_model_trunks.model.0.", ""): v for k, v in src_state_dict["model"].items() if "image_model_trunks.model.0." in k}, tgt_format="src")

        converted_target_encoder = convert_vit({k.replace("image_model_trunks.model.1.", ""): v for k, v in src_state_dict["model"].items() if "image_model_trunks.model.1." in k}, tgt_format="src")

        converted_predictor = convert_predictor({k.replace("heads.model.0.", ""): v for k, v in src_state_dict["model"].items() if "heads.model.0." in k}, tgt_format="src")

        # converted_predictor = convert_predictor(src_state_dict["predictor"], tgt_format="src")

        state_dict = {}
        state_dict.update({"encoder": converted_encoder})
        state_dict.update({"target_encoder": converted_target_encoder})
        state_dict.update({"predictor": converted_predictor})

    
    torch.save(state_dict, f"/cb/cold/aarti/ijepa/ijepa_convert_to_{convert_to}.mdl")


if __name__ == "__main__":
    src_ckpt_path = "/cb/cold/aarti/ijepa/IN1K-vit.h.14-300e.pth.tar"
    convert_ijepa_checkpoint(src_ckpt_path, convert_to="monolith")


    src_ckpt_path = "/cb/cold/aarti/ijepa/ijepa_convert_to_monolith.mdl"
    convert_ijepa_checkpoint(src_ckpt_path, convert_to="src")

    conv = torch.load("/cb/cold/aarti/ijepa/ijepa_convert_to_src.mdl")
    orig = torch.load("/cb/cold/aarti/ijepa/IN1K-vit.h.14-300e.pth.tar", map_location=torch.device('cpu'))

    main_keys = ["encoder", "target_encoder", "predictor"]

    for k in main_keys:
        orig_dict = orig[k]
        conv_dict = conv[k]

        assert len(orig_dict.keys()) == len(conv_dict.keys()), f"num_keys mismatch for {k}"

        for dict_k, orig_val in orig_dict.items():
            conv_val = conv_dict[dict_k]
            is_eq = torch.equal(orig_val, conv_val)

            if not is_eq:
                print(f"WARNING --- NOT EQUAL: {k}/{dict_k}")
            else:
                print(f"EQUAL: {k}/{dict_k}")