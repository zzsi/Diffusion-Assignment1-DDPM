# push to huggingface
from .dataset import AFHQDataModule
from PIL import Image

if __name__ == "__main__":
    user_name = "zzsi"
    # resolution = 64
    resolution = 512
    data_module = AFHQDataModule("data", 32, 4, -1, resolution, 1)
    train_ds = data_module.train_ds
    train_ds.transform = None
    val_ds = data_module.val_ds
    val_ds.transform = None
    print(train_ds[0][0].size)
    print(len(train_ds), len(val_ds))
    print(len(train_ds) + len(val_ds), "total images")

    # take a torch dataset and push to huggingface
    # https://huggingface.co/docs/huggingface_hub/v0.25.0/en/tutorials/push_to_hub
    from datasets import Dataset, DatasetDict
    # convert torch dataset to huggingface dataset
    def gen_train_ds():
        for img, label in train_ds:
            # resize to resolution x resolution
            if img.size != (resolution, resolution):
                img = img.resize((resolution, resolution), Image.Resampling.LANCZOS)
            yield {"image": img, "label": label}
    train_ds_hf = Dataset.from_generator(gen_train_ds)
    # Add both train and val datasets to the repo
    
    def gen_val_ds():
        for img, label in val_ds:
            if img.size != (resolution, resolution):
                img = img.resize((resolution, resolution), Image.Resampling.LANCZOS)
            yield {"image": img, "label": label}
    val_ds_hf = Dataset.from_generator(gen_val_ds)

    dataset_dict = DatasetDict({"train": train_ds_hf, "val": val_ds_hf})
    dataset_dict.push_to_hub(f"{user_name}/afhq{resolution}_16k", create_pr=False)
