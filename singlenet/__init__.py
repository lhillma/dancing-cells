from pathlib import Path


def get_dataset(
    dataset_path: Path, skip_first_n: int = 3, get_every: int = 1
) -> list[dict[str, Path]]:
    data_dicts = []
    run_folders = sorted([x for x in dataset_path.iterdir() if x.is_dir()])
    for run_folder in run_folders:
        vtk_files = sorted(run_folder.glob("**/*.vtk"), key=lambda x: int(x.stem.split("_")[-1]))

        for file in vtk_files:
            step_number = int(file.stem.split("_")[-1][:-3])

            if step_number < skip_first_n:
                continue
            if step_number % get_every != 0:
                continue

            data_dicts.append(
                {
                    "image": file,
                    "label": run_folder / "indices.txt",
                }
            )

    return data_dicts
