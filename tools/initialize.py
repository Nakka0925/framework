from pathlib import Path

import yaml


def init_settig(setting: Path) -> None:
    project_dir = setting.parent
    print(f"GENERATING SETTING FILE. -> {str(setting)}")

    template = {
        "destination": str(project_dir),
        "creature_data_destination" : "data/",
        "epochs" : 20,
        "batch_size" : 128,      
        "data_division" : "cross_val",
        "fold_num" : 5,
        ### graph or data name ###
        "accuracy_graph_name" : "",
        "loss_graph_name" : "",
        "heat_map_name" : "",
        "accuracy_loss_dataname" : "",
        "f1score_dataname" : "",
        "params_name" : ""
    }
    with open(setting, "w") as f:
        yaml.safe_dump(template, f)
    print("generating is done. Please rewrite if you need.")


def main() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    setting = project_dir / "train_setting.yml"
    if not setting.exists():
        init_settig(setting)


if __name__ == "__main__":
    main()