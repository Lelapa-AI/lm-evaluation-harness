import argparse

import yaml

tasks_lang = ['hausa', 'isizulu', 'swahili', 'xhosa', 'yoruba']
mt_lang = ['hau-eng', 'zul-eng', 'swa-eng', 'xho-eng', 'yor-eng']
mt_reverse = ['eng-hau', 'eng-zul', 'eng-swa', 'eng-xho', 'eng-yor']


def gen_lang_yamls(output_dir: str, overwrite: bool, mode: str, task: str, prompt: str) -> None:
    """
    Generate a yaml file for each language.

    :param output_dir: The directory to output the files to.
    :param overwrite: Whether to overwrite files if they already exist.
    """
    err = []
    languages = tasks_lang if mode is None else mt_lang if mode == "mmt" else mt_reverse
    for lang in languages:
        try:
            task_name = f"{task}_{prompt}_{lang}"
            yaml_template = f"{task}_{prompt}_default_yaml"

            file_name = f"{task_name}.yaml"
            with open(
                    f"{output_dir}/{file_name}", "w" if overwrite else "x", encoding="utf8"
            ) as f:
                f.write("# Generated by utils.py\n")
                yaml.dump(
                    {
                        "include": yaml_template,
                        "dataset_name": lang,
                        "task": f"{task_name}"
                    },
                    f,
                    allow_unicode=True,
                    width=float("inf"),
                )
        except FileExistsError:
            err.append(file_name)

    if len(err) > 0:
        raise FileExistsError(
            "Files were not created because they already exist (use --overwrite flag):"
            f" {', '.join(err)}"
        )


def main() -> None:
    """Parse CLI args and generate language-specific yaml files."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--overwrite",
        default=True,
        action="store_true",
        help="Overwrite files if they already exist",
    )
    parser.add_argument(
        "--output-dir", default="./mmt/native/english-african/", help="Directory to write yaml files to"
    )
    parser.add_argument(
        "--mode",
        default="mmt_reverse",
        choices=["mmt", "mmt_reverse"],
        help="Mode of task",
    )
    parser.add_argument(
        "--prompt",
        default="native",
        choices=["direct", "english", "native"],
        help="Prompt of the task",
    )
    parser.add_argument(
        "--task",
        default='mmt',
        choices=["mmt", "senti", "qa", "ner", "pos"],
        help="Task to create",
    )
    args = parser.parse_args()

    gen_lang_yamls(output_dir=args.output_dir, overwrite=args.overwrite, mode=args.mode, task=args.task, prompt=args.prompt)


if __name__ == "__main__":
    main()