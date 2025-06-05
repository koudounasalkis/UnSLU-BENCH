import torch
import os
import json
import copy
from datasets import load_dataset

import warnings
warnings.filterwarnings("ignore")

from unlearning_evaluation import print_evaluation_metrics
from utils import set_seed, define_model, read_data, parse_cmd_line_params
import unlearners

def main():
    args = parse_cmd_line_params()

    output_dir = f"results/{args.dataset}/{args.model_name_or_path.split('/')[-1]}"
    unlearner_name = args.unlearner if args.unlearner != "None" else "full"
    unlearner_name += "" if args.use_bad_teaching else "_light"
    unlearner_name += "_saliency" if args.saliency_map else ""
    output_dir = f"{output_dir}/{unlearner_name}_{args.lr}"
    output_dir = output_dir + f"_{args.epochs}/" if args.epochs > 1 else output_dir + "/"
    if os.path.exists(output_dir) and os.listdir(output_dir):
        print(f"Experiment {output_dir} already exists. Skipping...")
        return
    os.makedirs(output_dir, exist_ok=True)
    print("Output Directory: ", output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)

    print("Seed: ", args.seed)
    set_seed(args.seed)
    
    if args.dataset == "slurp" or args.dataset == "fsc":
        df_train, df_val, num_labels, label2id, id2label, labels = read_data(
            args.df_train, 
            args.df_val, 
            )
        print("Num labels: ", num_labels)

        _, df_test, _, _, _, _ = read_data(
        args.df_train, 
        args.df_test, 
        )
        
        _, df_retain, _, _, _, _ = read_data(
            args.df_train, 
            args.df_retain, 
            )
        
        _, df_forget, _, _, _, _ = read_data(
            args.df_train, 
            args.df_forget, 
            )

        ## Mapping intents to labels
        ## Model & Feature Extractor
        model_checkpoint = args.model_name_or_path
        feature_extractor, model = define_model(
            model_checkpoint,  
            num_labels, 
            label2id, 
            id2label, 
            args.feature_extractor_checkpoint,
            device
            )
        
        retain_dataset = dataset.Dataset_slurp_fsc(
            df_retain, 
            feature_extractor, 
            args.max_duration, 
            args.dataset
            )
        forget_dataset = dataset.Dataset_slurp_fsc(
            df_forget, 
            feature_extractor, 
            args.max_duration, 
            args.dataset
            )
        val_dataset = dataset.Dataset_slurp_fsc(
            df_val, 
            feature_extractor, 
            args.max_duration,
            dataset=args.dataset_name
            )
        test_dataset = dataset.Dataset_slurp_fsc(
            df_test, 
            feature_extractor, 
            args.max_duration,
            dataset=args.dataset_name,
            )
        
    elif args.dataset == "italic" or args.dataset == "de-DE" or args.dataset == "fr-FR":
        dataset = load_dataset("RiTA-nlp/ITALIC", args.dataset_name) if args.dataset == "italic" else load_dataset("FBK-MT/Speech-MASSIVE", args.dataset_name)

        ds_train = dataset["train"]
        ds_validation = dataset["validation"]
        if args.dataset == "italic":
            ds_test = dataset["test"] if args.dataset == "italic" else load_dataset("FBK-MT/Speech-MASSIVE-test", args.dataset_name, split="test")
        ds_forget, ds_retain = dataset.get_forget_retain_datasets(ds_train, f"{args.dataset_name}/")

        ## Mapping intents to labels
        intents = set(ds_train['intent'])
        with open(os.path.join(args.model_name_or_path, "config.json"), "r") as f:
            config = json.load(f)
            label2id = config["label2id"]
            id2label = config["id2label"]
            num_labels = len(id2label)
            print(label2id)

        ## Model & Feature Extractor
        feature_extractor, model = define_model(
            args.model_name_or_path, 
            num_labels, 
            label2id, 
            id2label, 
            args.feature_extractor_checkpoint
            )

        forget_dataset = dataset.Dataset_italic_sm(
            ds_forget, 
            feature_extractor, 
            label2id, 
            args.max_duration, 
            device
            )
        retain_dataset = dataset.Dataset_italic_sm(
            ds_retain, 
            feature_extractor, 
            label2id, 
            args.max_duration, 
            device
            )
        val_dataset = dataset.Dataset_italic_sm(
            ds_validation, 
            feature_extractor, 
            label2id, 
            args.max_duration, 
            device
            )
        test_dataset = dataset.Dataset_italic_sm(
            ds_test, 
            feature_extractor, 
            label2id, 
            args.max_duration, 
            device
            )

    time = 0

    retain_dataloader = torch.utils.data.DataLoader(retain_dataset, batch_size=args.batch, shuffle=True)
    forget_dataloader = torch.utils.data.DataLoader(forget_dataset, batch_size=args.batch, shuffle=True)

    # switch unlearners 
    if args.unlearner != "None":
        unlearner_name = args.unlearner
        model.to(device)
        switcher = {
            "finetune": unlearners.finetune,
            "cfk": unlearners.cf_k,
            "neggrad": unlearners.neggrad,
            "advancedneggrad": unlearners.advancedneggrad,
            "unsir": unlearners.unsir,
            "scrub": unlearners.scrub,
            "bad_teaching": unlearners.bad_teaching
        }
        unlearner = switcher.get(unlearner_name, None)
        if unlearner is None:
            print("Invalid unlearner")
            return

        # Esegui l'unlearner
        if unlearner_name == "unsir":
            time = unlearner(model, retain_dataloader, forget_dataloader, device, batch_size=args.batch, lr=args.lr, seed=args.seed)
        elif unlearner_name == "cfk":
            time = unlearner(model, retain_dataloader, forget_dataloader, device, lr=args.lr, unfreezed_encoder_layer=args.unfreeze_encoder_layer, seed=args.seed)
        elif unlearner_name == "bad_teaching":
            good_teacher = copy.deepcopy(model)
            if args.use_bad_teaching:
                print("Using bad teacher")
                _, bad_teacher = define_model(
                args.feature_extractor_checkpoint,
                num_labels, 
                label2id, 
                id2label, 
                device
                )
                bad_teacher.to(device)
            else:
                print("Not using bad teacher")
                bad_teacher = None
                unlearner_name = "bad_teaching_light"
            good_teacher.to(device)
            time = unlearner(model, bad_teacher, good_teacher, retain_dataset, forget_dataset, device, batch_size=args.batch, lr=args.lr, seed=args.seed)
        else:
            time = unlearner(model, retain_dataloader, forget_dataloader, device, lr=args.lr, seed=args.seed, num_epochs=args.epochs)

    dict = print_evaluation_metrics(model, forget_dataset, val_dataset, test_dataset, output_dir, device, save=True)
    txt_dir = f"{output_dir}/evaluation_metrics.txt" if args.seed == 0 else f"{output_dir}/evaluation_metrics_seed_{args.seed}.txt"

    with open(txt_dir, 'w') as f:
        for key, value in dict.items():
            f.write(f'{key}: {value}\n')
        f.write(f'Unlearning Time: {time}\n')

    return


if __name__ == "__main__":
    main()