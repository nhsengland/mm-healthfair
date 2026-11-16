import argparse
import glob
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import shap
import toml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import CollateFn, CollateTimeSeries, MIMIC4Dataset
from models import MMModel
from utils.functions import load_pickle, save_pickle
from utils.preprocessing import encode_categorical_features
from utils.shap_utils import (
    aggregate_ts,
    estimate_mm_summary,
    get_feature_names,
    get_shap_local_decision_plot,
    get_shap_summary_plot,
    get_shap_values,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multimodal SHAP feature importance analysis."
    )
    parser.add_argument(
        "data_path",
        type=str,
        help="Path to the pickled data dictionary generated from prepare_data.py.",
        default="../outputs/prep_data/mmfair_feat.pkl",
    )
    parser.add_argument(
        "--col_path",
        "-p",
        type=str,
        help="Path to the pickled column dictionary generated from prepare_data.py.",
        default="../outputs/prep_data/mmfair_cols.pkl",
    )
    parser.add_argument(
        "--feat_names",
        "-f",
        type=str,
        help="Path to a JSON file containing lookup names for each feature.",
        default="../config/shap_feat_map.json",
    )
    parser.add_argument(
        "--ids_path",
        "-i",
        type=str,
        help="Directory containing test ids.",
        default="../outputs/prep_data",
    )
    parser.add_argument(
        "--exp_path",
        "-x",
        type=str,
        help="Directory to store explanation plots.",
        default="../outputs/explanations",
    )
    parser.add_argument(
        "--eval_path",
        "-e",
        type=str,
        help="Evaluation path for obtaining risk dictionary as .pkl file (eval_path+model_path).",
        default="../outputs/evaluation",
    )
    parser.add_argument(
        "--attr_path",
        "-a",
        type=str,
        help="Directory containing attributes metadata (original ehr_static.csv).",
        default="../outputs/ext_data/ehr_static.csv",
    )
    parser.add_argument(
        "--outcome",
        "-o",
        type=str,
        help="Binary outcome to use for multimodal learning (one of the labels in targets.toml)."
        "Defaults to prediction of in-hospital death.",
        default="in_hosp_death",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        help="Directory containing the saved model metadata.",
        default="logs/nhs-mm-healthfair",
    )
    parser.add_argument(
        "--model_path",
        "-m",
        type=str,
        help="Directory pointing to the saved model checkpoint (must be inside model_dir).",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="../config/model.toml",
        help="Path to config toml file containing model training parameters.",
    )
    parser.add_argument(
        "--targets",
        "-t",
        type=str,
        default="../config/targets.toml",
        help="Path to config toml file containing lookup fields and outcomes.",
    )
    parser.add_argument(
        "--exp_mode",
        type=str,
        default="global",
        help="Use global or local explanations. If global, uses DeepExplainer over static and timeseries modalities in the test set."
        "If local, uses DeepExplainer to explain individual predictions over all modalities (requires a fused static-timeseries-text model).",
    )
    parser.add_argument(
        "--use_heatmaps",
        action="store_true",
        help="Use heatmaps for SHAP summary plots instead of beeswarm plots.",
    )
    parser.add_argument(
        "--global_max_features",
        "-g",
        type=int,
        default=20,
        help="Top N features to plot in global-level explanations.",
    )
    parser.add_argument(
        "--local_risk_group",
        "-r",
        type=int,
        default=10,
        help="Risk quantile to use for local-level explanations (generated in evaluate.py).",
    )
    parser.add_argument(
        "--notes_offset_ref",
        action="store_true",
        help="Offset SHAP colormap center for local-level text plot using the expected SHAP value (batch-wise mean)."
        "Defaults to False (SHAP colormap centered around 0).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        dest="verbose",
        action="store_true",
        default=False,
        help="Control verbosity.",
    )
    args = parser.parse_args()
    config = toml.load(args.config)
    targets = toml.load(args.targets)
    ### General setup
    outcomes = targets["outcomes"]["labels"]
    outcomes_disp = targets["outcomes"]["display"]
    outcomes_col = targets["outcomes"]["colormap"]
    attributes = targets["attributes"]["labels"]
    attr_disp = targets["attributes"]["display"]
    batch_size = 16
    ### Edit when adding more timeseries measurements
    ts_types = ["ts-vitals", "ts-labs"]
    ### Local-level setup for extracting risk profile
    tg_gender = targets["shap_profile"]["sex"]
    tg_ms = targets["shap_profile"]["marital_status"]
    tg_eth = targets["shap_profile"]["ethnicity"]
    tg_insurance = targets["shap_profile"]["insurance"]
    ### Model-specific setup
    model_path = os.path.join(
        os.path.join(args.model_dir, args.model_path), args.model_path + ".ckpt"
    )
    risk_path = os.path.join(
        os.path.join(args.eval_path, args.model_path), "pf_" + args.model_path + ".pkl"
    )
    exp_path = os.path.join(args.exp_path, args.model_path)
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    num_workers = config["data"]["num_workers"]
    ### Infer fusion method and modalities from model_path
    modalities = []
    for arg in args.model_path.split("_"):
        if "static" in arg:
            modalities.append("static")
        elif "timeseries" in arg:
            modalities.append("timeseries")
        elif "notes" in arg:
            modalities.append("notes")
    static_only = True if (len(modalities) == 1) and ("static" in modalities) else False
    with_notes = True if "notes" in modalities else False
    fusion_method = "None"
    if "mag" in args.model_path:
        fusion_method = "IF-mag"
    elif "concat" in args.model_path:
        fusion_method = "IF-concat"
    if args.outcome not in outcomes:
        print(f"Outcome {args.outcome} must be included in targets.toml.")
        sys.exit()
    outcome_idx = outcomes.index(args.outcome)
    compute_shap = False
    ### Set plotting style
    plt.rcParams.update(
        {"font.size": 13, "font.weight": "normal", "font.family": "serif"}
    )
    print("------------------------------------------")
    print("MMHealthFair: Multimodal SHAP feature importance analysis")
    print("------------------------------------------")
    print(f'Evaluating feature importances for outcome "{outcomes_disp[outcome_idx]}"')
    print(f"Modalities used: {modalities}")
    print(f"Fusion method: {fusion_method}")
    print(f"Mode: {args.exp_mode}")
    if args.exp_mode == "global":
        print(f"Global max features: {args.global_max_features}")
    else:
        print(f"Risk group target: {args.local_risk_group}")
    print("------------------------------------------")
    # Get test ids
    if (
        len(
            glob.glob(
                os.path.join(args.ids_path, "testing_ids_" + args.outcome + ".csv")
            )
        )
        == 0
    ):
        print(f"No test ids found for outcome {args.outcome}. Exiting..")
        sys.exit()

    if (len(model_path) == 0) and (not args.group_models):
        print(f"No model found at {args.model_path}. Exiting..")
        sys.exit()

    if not os.path.exists(risk_path):
        print(f"No risk dictionary found at {risk_path}. Exiting..")
        sys.exit()

    if not os.path.exists(args.feat_names):
        print(f"No feature names found at {args.feat_names}. Exiting..")
        sys.exit()

    if not os.path.exists(
        os.path.join(args.exp_path, f"{args.model_path}/shap_{args.model_path}.pkl")
    ):
        print(
            f"No SHAP dictionary found at {os.path.join(args.exp_path, f'{args.model_path}/shap_{args.model_path}.pkl')}."
        )
        print("Running batch-wise SHAP computation...")
        compute_shap = True

    if not compute_shap:
        print(
            f"Loading pre-computed SHAP dictionary from {os.path.join(args.exp_path, f'shap_{args.model_path}.pkl')}"
        )
        shap_dict = load_pickle(
            os.path.join(args.exp_path, f"{args.model_path}/shap_{args.model_path}.pkl")
        )

    test_ids = (
        pl.read_csv(os.path.join(args.ids_path, "testing_ids_" + args.outcome + ".csv"))
        .select("subject_id")
        .to_numpy()
        .flatten()
    )
    risk_dict = load_pickle(risk_path)
    emb_dict = load_pickle(args.data_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_set = MIMIC4Dataset(
        args.data_path,
        args.col_path,
        "test",
        ids=test_ids,
        static_only=static_only,
        with_notes=with_notes,
    )
    test_set.print_label_dist()
    ### Load model
    model = MMModel.load_from_checkpoint(checkpoint_path=model_path)
    dataloader = DataLoader(
        test_set,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=CollateFn() if static_only else CollateTimeSeries(),
        persistent_workers=True if num_workers > 0 else False,
        shuffle=False,  # Ensure no shuffling to reproduce exact test set samples
    )
    ### SHAP setup
    ### Get names as recorded in SHAP dictionary
    shap_colnames = get_feature_names(test_set, modalities=modalities)
    ### Load JSON file mapping feature names to display names
    with open(args.feat_names) as f:
        feature_names = json.load(f)
    ### Pre-compute SHAP values if not already done
    if compute_shap:
        shap_dict = {}
        for batch_idx, batch in tqdm(
            enumerate(dataloader),
            desc="Computing SHAP values across all test batches...",
        ):
            shap_scores = get_shap_values(
                model, batch, device=device, num_ts=2, modalities=modalities
            )
            shap_dict["batch_" + str(batch_idx)] = shap_scores

        save_pickle(
            shap_dict, args.exp_path, f"{args.model_path}/shap_{args.model_path}.pkl"
        )
        print(
            f"SHAP values saved to {os.path.join(args.exp_path, f'shap_{args.model_path}.pkl')}"
        )
    ### Collect feature explanations
    if args.exp_mode == "global":
        print("Displaying multimodal SHAP summary plots...")
        if "static" in modalities:
            target_names = [feature_names.get(k, k) for k in shap_colnames["static"]]
            # Collect all static SHAP values into a unified np.array
            shap_global_values = []
            actual_values = []
            for batch_idx, batch in enumerate(dataloader):
                for i in range(len(batch[0])):
                    batch_shap = shap_dict["batch_" + str(batch_idx)]["static"][
                        i
                    ].reshape(1, -1)
                    shap_global_values.extend(np.array(batch_shap))
                    actual_values.extend(np.array(batch[0][i]))
            shap_global_values = np.array(shap_global_values)
            actual_values = np.array(actual_values)
            shap_obj = shap.Explanation(
                values=shap_global_values,
                feature_names=target_names,
                data=actual_values,
            )

            get_shap_summary_plot(
                shap_obj,
                fusion_type=fusion_method,
                modality="static",
                outcome=outcomes_disp[outcome_idx],
                max_features=args.global_max_features,
                save_path=os.path.join(
                    exp_path,
                    f"shap_global_{outcomes[outcome_idx]}_static_summary.png",
                ),
                heatmap=args.use_heatmaps,
            )

        if "timeseries" in modalities:
            ### Replace if adding more timeseries measurements
            for i in range(len(ts_types)):
                print(f"Processing SHAP values for {ts_types[i]}...")
                if ts_types[i] == "ts-vitals":
                    target_names = shap_colnames["ts-vitals"]
                    tg_field = "dynamic0"
                else:
                    target_names = [
                        feature_names.get(k, k) for k in shap_colnames["ts-labs"]
                    ]
                    tg_field = "dynamic1"
                # Collect all timeseries SHAP values into a unified np.array
                shap_global_values = []
                actual_values = []
                test_id = 0
                for batch_idx, batch in enumerate(dataloader):
                    for ts_i in range(len(batch[2][i])):
                        batch_shap = shap_dict["batch_" + str(batch_idx)]["timeseries"][
                            tg_field
                        ][ts_i]
                        agg_shap = np.mean(np.mean(batch_shap, axis=2), axis=0).reshape(
                            1, -1
                        )
                        test_id += 1
                        # Only average over valid (non -1, 0) values in batch[2][i]
                        data = np.array(batch[2][i][ts_i])
                        agg_values = aggregate_ts(data)
                        # Append values
                        shap_global_values.extend(np.array(agg_shap))
                        actual_values.extend(np.array(agg_values))
                shap_global_values = np.array(shap_global_values)
                actual_values = np.array(actual_values)
                shap_obj = shap.Explanation(
                    values=shap_global_values,
                    feature_names=target_names,
                    data=actual_values,
                )
                get_shap_summary_plot(
                    shap_obj,
                    fusion_type=fusion_method,
                    modality="timeseries",
                    outcome=outcomes_disp[outcome_idx],
                    max_features=args.global_max_features,
                    save_path=os.path.join(
                        exp_path,
                        f"shap_global_{outcomes[outcome_idx]}_{ts_types[i]}_summary.png",
                    ),
                    heatmap=args.use_heatmaps,
                )

        if "notes" in modalities:
            # Collect all static SHAP values into a unified np.array
            shap_global_values = []
            actual_values = []
            notes_test = []
            feature_names = []
            notes_ctr = 0
            emb_max = 768
            emb_min = 3
            for i in range(len(test_ids)):
                ### Collect raw notes from test samples
                data_notes = np.array([s[0] for s in emb_dict[test_ids[i]]["notes"]])
                data_notes_tr = []
                for j in range(len(data_notes)):
                    ## Trim to max embedding length for BioBERT
                    if len(data_notes[j]) > emb_max:
                        data_notes_tr.append(data_notes[j][:768])
                    elif len(data_notes[j]) < emb_min:
                        continue
                    else:
                        data_notes_tr.append(data_notes[j])
                ### Shorten notes for display purposes
                notes_test.append(np.array(data_notes_tr))
                feature_names.extend(
                    np.array(
                        [
                            data_notes_tr[j][:75] + "..."
                            for j in range(len(data_notes_tr))
                        ]
                    )
                )

            for batch_idx, _ in enumerate(dataloader):
                for shap_v in shap_dict["batch_" + str(batch_idx)]["notes"]:
                    ### Need to retrieve correct length from original clinical note and filter out batch-wise padded SHAP values
                    batch_shap = np.array(
                        shap_v[0][: len(notes_test[notes_ctr])]
                    ).reshape(1, -1)[0]
                    if len(batch_shap) < emb_min:
                        continue
                    ### Flatten batch_shap and add to global values
                    shap_global_values.extend(np.array(batch_shap))
                    notes_ctr += 1

            shap_global_values = np.array(shap_global_values)
            feature_names = np.array(feature_names)
            shap_obj = shap.Explanation(
                values=shap_global_values, feature_names=feature_names, data=None
            )

            get_shap_summary_plot(
                shap_obj,
                fusion_type=fusion_method,
                modality="notes",
                outcome=outcomes_disp[outcome_idx],
                max_features=args.global_max_features,
                save_path=os.path.join(
                    exp_path,
                    f"shap_global_{outcomes[outcome_idx]}_notes_summary.png",
                ),
                heatmap=args.use_heatmaps,
            )

    if args.exp_mode == "local":
        print(
            f"Generating local-level SHAP explanations for random subject at risk quantile {args.local_risk_group}..."
        )
        if "p_id" not in shap_dict["batch_0"].keys():
            print(
                "No patient IDs found in SHAP dictionary. Matching IDs to SHAP values..."
            )
            p_ctr = 0
            for i in range(len(shap_dict)):
                ids_list = []
                ctr_list = []
                for _ in shap_dict["batch_" + str(i)]["static"]:
                    ids_list.append(test_ids[p_ctr])
                    ctr_list.append(p_ctr)
                    p_ctr += 1
                shap_dict["batch_" + str(i)]["p_id"] = ids_list
                shap_dict["batch_" + str(i)]["ctr"] = ctr_list

        shap_colnames = get_feature_names(test_set, modalities=modalities)
        ### Get static SHAP values for patient
        static_names = [feature_names.get(k, k) for k in shap_colnames["static"]]
        lab_names = [feature_names.get(k, k) for k in shap_colnames["ts-labs"]]
        vitals_names = shap_colnames["ts-vitals"]
        ### Load in unscaled test data
        ehr_static = pl.read_csv(args.attr_path)
        static_values = encode_categorical_features(pl.DataFrame(ehr_static))
        ### Extract selected patient profiles
        static_values = static_values.filter(
            (
                (pl.col("gender_F") == 1)
                if tg_gender == "F"
                else (pl.col("gender_F") == 0)
            )
            & (pl.col("marital_status_" + str(tg_ms)) == 1)
            & (pl.col("race_group_" + str(tg_eth)) == 1)
            & (pl.col("insurance_" + str(tg_insurance)) == 1)
        )
        # Get a random subject with the specified risk quantile and attributes
        risk_idx = pd.DataFrame(
            {k: v for k, v in risk_dict.items() if k in ["test_ids", "risk_quantile"]}
        )
        risk_idx = risk_idx[risk_idx["risk_quantile"] == args.local_risk_group]
        risk_idx = risk_idx[risk_idx["test_ids"].isin(static_values["subject_id"])]
        risk_idx = risk_idx[risk_idx["test_ids"].isin(test_ids)]
        if len(risk_idx) == 0:
            print(
                f"No patients found with risk quantile {args.local_risk_group} with specified attributes."
            )
            print("Please provide a different risk quantile or attributes.")
            sys.exit()
        ### Randomly sample an individual case
        risk_idx = risk_idx.sample(n=1, random_state=42)["test_ids"].to_numpy().tolist()
        ### Set specific patient ID for testing here
        # risk_idx = []
        print(f"Selected patient ID: {risk_idx[0]}")
        static_values = static_values.filter(
            pl.col("subject_id").is_in(list(map(int, risk_idx)))
        )
        static_values = (
            static_values.select(shap_colnames["static"]).to_pandas().to_numpy()
        )
        ### Get multimodal SHAP values for single patient
        ctr = 0
        for sb in shap_dict.keys():
            ctr += len(shap_dict[sb]["p_id"])
            if risk_idx[0] not in shap_dict[sb]["p_id"]:
                continue
            ### Collect Static SHAP values
            pt_idx = shap_dict[sb]["p_id"].index(risk_idx[0])
            static_shap = shap_dict[sb]["static"][pt_idx].reshape(1, -1)
            shap_expected = shap_dict[sb]["static_expected"][0]
            shap_static_values = np.array(static_shap)[0]
            ### Collect and aggregate Timeseries SHAP values
            shap_vitals = np.array(shap_dict[sb]["timeseries"]["dynamic0"][pt_idx])
            shap_vitals_expected = np.array(
                shap_dict[sb]["timeseries"]["dynamic0_expected"]
            )[0]
            shap_labs = np.array(shap_dict[sb]["timeseries"]["dynamic1"][pt_idx])
            shap_labs_expected = np.array(
                shap_dict[sb]["timeseries"]["dynamic1_expected"]
            )[0]
            ### Collect notes SHAP values
            data_notes = np.array([s[0] for s in emb_dict[risk_idx[0]]["notes"]])
            ### Need to retrieve correct length from original clinical note and filter out batch-wise padded SHAP values
            shap_notes = np.array(
                shap_dict[sb]["notes"][pt_idx][0][: len(data_notes)]
            ).reshape(1, -1)[0]
            shap_notes_expected = np.array(shap_dict[sb]["notes_expected"])[0]
            ### Need to extract SHAP scores from final hidden layer due to LSTM setup
            ## Here to compute the TS-SHAP values we average pool the SHAP values across hidden units, then sum the importances over each timestep.
            ## This is an arbitrary approach to generate a single list of SHAP values, there may be other more robust ways.
            agg_vitals = np.sum(np.mean(shap_vitals, axis=2), axis=0).reshape(1, -1)[0]
            agg_labs = np.sum(np.mean(shap_labs, axis=2), axis=0).reshape(1, -1)[0]
            # mm_vitals = np.mean(shap_vitals, axis=2).reshape(1, -1)[0]
            # mm_labs = np.mean(shap_labs, axis=2).reshape(1, -1)[0]
            pt_batch_idx = int(sb.split("_")[1])
            # Get original data
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx != pt_batch_idx:
                    continue
                data_vitals = np.array(batch[2][0][pt_idx])
                data_labs = np.array(batch[2][1][pt_idx])
                data_vitals = np.array(aggregate_ts(data_vitals))
                data_labs = np.array(aggregate_ts(data_labs))

        shap_mm_scores, shap_expected_ovr, shap_max_ovr, shap_min_ovr = (
            estimate_mm_summary(
                [shap_static_values, agg_vitals, agg_labs, shap_notes],
                [
                    shap_expected,
                    shap_vitals_expected,
                    shap_labs_expected,
                    shap_notes_expected,
                ],
            )
        )

        # Generate Explanation objects
        shap_static_obj = shap.Explanation(
            values=np.round(shap_static_values, 5),
            base_values=shap_expected_ovr,
            feature_names=static_names,
            data=static_values,
        )
        shap_vitals_obj = shap.Explanation(
            values=np.round(agg_vitals, 5),
            base_values=shap_expected_ovr,
            feature_names=vitals_names,
            data=data_vitals,
        )
        shap_labs_obj = shap.Explanation(
            values=np.round(agg_labs, 5),
            base_values=shap_expected_ovr,
            feature_names=lab_names,
            data=data_labs,
        )
        shap_notes_obj = shap.Explanation(
            values=np.round(shap_notes, 5),
            base_values=shap_expected_ovr,
            feature_names=None,
            data=data_notes,
        )

        get_shap_local_decision_plot(
            [shap_static_obj, shap_vitals_obj, shap_labs_obj, shap_notes_obj],
            risk_quantile=args.local_risk_group,
            save_static_path=os.path.join(
                exp_path,
                f"shap_local_{outcomes[outcome_idx]}_static_decision.png",
            ),
            save_ts0_path=os.path.join(
                exp_path,
                f"shap_local_{outcomes[outcome_idx]}_tsv_decision.png",
            ),
            save_ts1_path=os.path.join(
                exp_path,
                f"shap_local_{outcomes[outcome_idx]}_tsl_decision.png",
            ),
            save_nt_path=os.path.join(
                exp_path,
                f"shap_local_{outcomes[outcome_idx]}_notes_text.png",
            ),
            shap_range=(shap_min_ovr, shap_max_ovr),
            mm_scores=shap_mm_scores,
        )
