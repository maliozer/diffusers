import datasets
import os
from PIL import Image
import csv
from ast import literal_eval

class CustomImageCaptioningDataset(datasets.GeneratorBasedBuilder):
    def __init__(self, csv_file='/content/output.csv', image_folder='/content/pororo_diff_finetune_dataset2', **kwargs):
        super().__init__(**kwargs)
        self.csv_file = csv_file
        self.image_folder = image_folder

    def _info(self):
        return datasets.DatasetInfo(
            description="Custom Image Captioning Dataset",
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "caption": datasets.Value("string"),
                    "prev_img": datasets.Sequence(datasets.Value("string")),
                    "image_name": datasets.Value("string"),
                }
            ),
            supervised_keys=("image", "caption"),
            homepage="https://example.com/dataset",
            citation="Author, A. (Year). Title of the Dataset. Publisher.",
            license="CC BY-SA 4.0",
        )

    def _split_generators(self, dl_manager):
            # Load the CSV file containing (filename, text) pairs
            with open(self.csv_file, "r") as csv_file:
                reader = csv.DictReader(csv_file)
                caption_data = [(row["file_name"], row["text"], literal_eval(row["prev_img"])) for row in reader]

            # Create a single split for the entire dataset
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={"caption_data": caption_data},
                ),
            ]

    def _generate_examples(self, caption_data):
        for idx, (file_name, caption, prev_text) in enumerate(caption_data):
            # Construct the full image path
            image_path = os.path.join(self.image_folder, file_name)

            # Load the image using PIL
            with open(image_path, "rb") as img_file:
                image = Image.open(img_file)
                image_bytes = img_file.read
            yield idx, {"image": {"path": image_path, "bytes": image_bytes}, "caption": caption, "prev_img": prev_text, "image_name":image_path}
